import os
import numpy as np
from PIL import Image
from scipy.misc import imread

import torch
import torch.utils.data as data
from torchvision.ops.boxes import clip_boxes_to_image


class SequentialGetitem(data.Dataset):

    def __init__(self, roidb, num_classes, div=False, BGR=True):
        self.roidb = roidb
        self._num_classes = num_classes
        self.div = div
        self.BGR = BGR

    def get_height_and_width(self, index):
        return self.roidb[index]['height'], self.roidb[index]['width']

    def __getitem__(self, index):
        single_item = self.roidb[index]
        # Image
        im = imread(single_item['img_path'])  # RGB, HWC, 0-255
        # im = np.array(Image.open(single_item['img_path']).convert('RGB'))
        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.concatenate((im, im, im), axis=2)

        # divided by 255 for PyTorch pre-trained model
        if self.div:
            im = im / 255.
        # flip the channel, RGB to BGR for caffe pre-trained model
        if self.BGR:
            im = im[:, :, ::-1]
        if single_item['flipped']:
            im = im[:, ::-1, :]
        im = im.astype(np.float32, copy=False)
        image = torch.from_numpy(im).permute(2, 0, 1)  # HWC to CHW

        # Targets
        gt_boxes = single_item['boxes'].astype(np.int32, copy=False)
        gt_boxes = torch.from_numpy(gt_boxes).float()  # TODO(BUG): dtype
        # clip boxes of which coordinates out of the image resolution
        gt_boxes = clip_boxes_to_image(gt_boxes, tuple(image.shape[1:]))

        target = dict(
            boxes=gt_boxes,  # (num_boxes 4)
            labels=torch.from_numpy(single_item['gt_classes']),
            pids=torch.from_numpy(single_item['gt_pids']),
            img_name=single_item['img_name'],
        )
        if 'mask_path' in single_item:
            # foreground mask
            mask = imread(single_item['mask_path']).astype(np.int32, copy=False)  # (h w) in {0,255}
            assert np.ndim(mask) == 2
            if single_item['flipped']: mask = mask[:, ::-1]
            target['mask'] = torch.from_numpy(mask.copy())[None] / 255.  # 3D tensor(1HW) in {0,1}

        item = dict(image=image, target=target)

        # visualization
        # util.plot_gt_on_img([image], [target], write_path=
        # "/home/caffe/code/deep-person-search/cache/img_with_gt_box/gt%d.jpg" % np.random.choice(list(range(10)), 1))

        return item

    def __len__(self):
        return len(self.roidb)


class ProbeGetitem(data.Dataset):
    def __init__(self, probe_list):
        self.probes = probe_list

    def __getitem__(self, index):
        outputs = self.probes[index]
        im_name, roi = outputs[0], outputs[1]
        # process input image
        im = imread(im_name)  # RGB, HWC, 0-255
        # im = np.array(Image.open(im_name).convert('RGB'))  # RGB, HWC, 0-255
        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.concatenate((im, im, im), axis=2)
        # flip the channel, since the original one using cv2, rgb -> bgr
        im = im[:, :, ::-1]  # TODO(NOTE): RGB to BGR for caffe pretrained model
        im = im.astype(np.float32, copy=False)
        image = torch.from_numpy(im).permute(2, 0, 1)  # HWC to CHW
        box = torch.from_numpy(roi.reshape(1, 4)).float()  # shape (1 4)
        item = dict(image=image, target=box)
        return item

    def __len__(self):
        return len(self.probes)


class ListImageLoader(data.Dataset):
    def __init__(self, path):
        self.path = path
        self.img_list = [os.path.join(self.path, p) for p in os.listdir(path)]

    def __getitem__(self, index):
        img_path = self.img_list[index]
        # read image
        im = imread(img_path)  # PIL image, RGB, HWC
        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.concatenate((im, im, im), axis=2)
        # flip the channel, since the original one using cv2, rgb -> bgr
        im = im[:, :, ::-1]  # NOTE: RGB to BGR for caffe pretrained model

        im = im.astype(np.float32, copy=False)
        image = torch.from_numpy(im).permute(2, 0, 1)  # HWC to CHW
        return image

    def __len__(self):
        return len(self.img_list)
