import random
import math
import torch
from torch import nn

from torchvision.ops import misc as misc_nn_ops
from lib.head.RoIHeads import paste_masks_in_image


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)


class GeneralizedRCNNTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """

    def __init__(self, min_size, max_size, image_mean, image_std):
        super(GeneralizedRCNNTransform, self).__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std

    def forward(self, images, targets=None):
        for i in range(len(images)):
            image = images[i]
            if image.dim() == 4 and image.size(0) == 1:  # todo-note: add for 4D input
                image = image.squeeze(0)
            target = targets[i] if targets is not None else targets
            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            image = self.normalize(image)
            image, target = self.resize(image, target)
            images[i] = image
            if targets is not None:
                targets[i] = target
        image_sizes = [img.shape[-2:] for img in images] # image size after resizing
        images = self.batch_images(images) # image size after padding
        image_list = ImageList(images, image_sizes) # assign resized sizes

        ##### Added masks, 20200402
        if targets is not None and 'mask' in targets[0]:
            masks = self.batch_masks([t['mask'] for t in targets])  # 4D (N1HW)
            for idx, t in enumerate(targets): t['mask'] = masks[idx]
        #####

        return (image_list, targets) if targets is not None else image_list

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        h, w = image.shape[-2:]  # original size
        min_size = float(min(image.shape[-2:]))
        max_size = float(max(image.shape[-2:]))
        if self.training:
            size = random.choice(self.min_size)
        else:
            # FIXME assume for now that testing uses the largest scale
            size = self.min_size[-1]
        scale_factor = size / min_size
        if max_size * scale_factor > self.max_size:
            scale_factor = self.max_size / max_size
        image = torch.nn.functional.interpolate(
            image[None], scale_factor=scale_factor, mode='bilinear', align_corners=False)[0]

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        if "mask" in target:
            mask = target["mask"]  # 3D tensor (1HW)
            mask = misc_nn_ops.interpolate(mask[None].float(), scale_factor=scale_factor)[0].byte()  # 3D tensor
            target["mask"] = mask

        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = resize_keypoints(keypoints, (h, w), image.shape[-2:])
            target["keypoints"] = keypoints
        return image, target

    def batch_images(self, images, size_divisible=32):
        # concatenate
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))

        stride = size_divisible
        max_size = list(max_size)
        max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
        max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
        max_size = tuple(max_size)

        batch_shape = (len(images),) + max_size
        batched_imgs = images[0].new(*batch_shape).zero_()
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs  # 4D (NCHW)

    def batch_masks(self, masks, size_divisible=32):
        # concatenate
        max_size = tuple(max(s) for s in zip(*[img.shape for img in masks]))

        stride = size_divisible
        max_size = list(max_size)
        max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
        max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
        max_size = tuple(max_size)

        batch_shape = (len(masks),) + max_size
        batched_imgs = masks[0].new(*batch_shape).zero_()
        for img, pad_img in zip(masks, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs  # 4D (NCHW)

    def postprocess(self, result, image_shapes, original_image_sizes):
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks
            if "keypoints" in pred:
                keypoints = pred["keypoints"]
                keypoints = resize_keypoints(keypoints, im_s, o_im_s)
                result[i]["keypoints"] = keypoints
        return result


def resize_keypoints(keypoints, original_size, new_size):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
    ratio_h, ratio_w = ratios
    resized_data = keypoints.clone()
    resized_data[..., 0] *= ratio_w
    resized_data[..., 1] *= ratio_h
    return resized_data


def resize_boxes(boxes, original_size, new_size):
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)
