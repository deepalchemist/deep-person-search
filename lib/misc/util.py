import os
import json
import cv2
import time
import math
import errno
import random
import numpy as np
from PIL import Image
from thop import profile
from scipy.misc import imread
from os.path import dirname as ospdn

import torch
from torchvision.ops import boxes as ops_box

"""##########
### Train ###
##########"""


def init_random_seed(seed):
    random.seed(seed)  # python
    np.random.seed(seed)  # numpy
    torch.manual_seed(seed)  # lead to torch.initial_seed==opt.seed, torch CPU
    torch.cuda.manual_seed(seed)  # torch GPU
    torch.cuda.manual_seed_all(seed)  # torch GPU
    torch.backends.cudnn.benchmark = False  # accelerate computing
    torch.backends.cudnn.deterministic = True  # avoid inference performance variation
    # torch.backends.cudnn.enabled = True


def init_device(gpu_ids):
    th_devices = [torch.device("cuda:{}".format(gpu)) for gpu in gpu_ids]
    DEVICE = th_devices[0]
    return th_devices, DEVICE


def filter_required_pids(batch_pids, batch_props=None, required_pid=-1):
    """
    Args:
        batch_pids (list): list of (n_roi_per_img) tensor
        batch_props (list): list of (n_roi_per_img 4) tensor
    Returns:
    """

    if batch_props is not None:
        assert len(batch_props) == len(batch_pids)

    keeps = [pid >= required_pid for pid in batch_pids]
    valid_pids = [pid[k] for pid, k in zip(batch_pids, keeps)]
    if batch_props is not None:
        batch_props = [prop[k] for prop, k in zip(batch_props, keeps)]

    return keeps, valid_pids, batch_props


def plot_gt_on_img(im_batch, target, write_path=None):
    ''' Visualize a mini-batch for debugging.

    Args:
        im_batch (list of 3D tensor): (3 H W) BGR 0-255
        target (list of dict): (bs MAX_NUM_GT_BOXES 6)

    Returns:

    '''
    pixel_means = [102.9801, 115.9465, 122.7717]
    pixel_means = torch.tensor(pixel_means, dtype=torch.float)[:, None, None]

    batch_size = len(im_batch)
    batch_cv2im = []
    for ii in range(batch_size):
        rec_im = im_batch[ii].detach().cpu() + pixel_means
        cv2im = rec_im.permute(1, 2, 0).numpy()  # (H W 3)
        for (x1, y1, x2, y2), is_person, pid in zip(target[ii]['boxes'], target[ii]['labels'], target[ii]['pids'], ):
            if is_person != 1:
                continue
            ec = (255, 0, 0) if pid == -1 else (0, 255, 0)
            if pid != -1:
                cv2im = cv2.rectangle(cv2im, (x1, y1 - 16), (x2, y1), ec,
                                      thickness=-1)  # thickness=negative means filled rectangle
                cv2im = cv2.putText(cv2im, str(int(pid)), org=(x1 + 4, y1 - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            cv2im = cv2.rectangle(cv2im, (x1, y1), (x2, y2), ec, thickness=1)

        batch_cv2im.append(cv2im)
    batch_tensor_im = [torch.as_tensor(im.get()) for im in batch_cv2im]  # HWC
    # batching images with diff size
    ret = batching_images_by_padding(batch_tensor_im)
    ret = torch.cat(ret.split(1, dim=0), dim=1).squeeze()
    ret = ret.numpy()
    # import pdb; pdb.set_trace()
    # ret = cv2.resize(ret, dsize=(2000, 1000))
    if write_path is not None:
        cv2.imwrite(write_path, ret)
    return ret


def batching_images_by_padding(batch_images, size_divisible=32):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in batch_images]))  # list of 3D tensor

    stride = size_divisible
    max_size = list(max_size)
    max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
    max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
    max_size = tuple(max_size)

    batch_shape = (len(batch_images),) + max_size
    batched_imgs = batch_images[0].new(*batch_shape).zero_()
    for img, pad_img in zip(batch_images, batched_imgs):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    return batched_imgs


def vis_masks(masks):
    """
    Args:
        masks: list of 3D tensor (1 h w)
    """
    batched_masks = torch.stack(masks, dim=0)
    # batched_masks = batching_images_by_padding(masks)
    batched_masks = torch.cat([batched_masks] * 3, dim=1) * 255  # (n 3 h w)
    return batched_masks.cpu().float()


def vis_gt_boxes(im_batch, target, pixel_mean, query_pid=None):
    ''' Visualize a mini-batch for debugging.

    Args:
        im_batch (list of 3D tensor): (3 H W) BGR
        target (list of dict): (bs MAX_NUM_GT_BOXES 6)

    Returns:

    '''
    batch_size = len(im_batch)
    pixel_means = torch.tensor(pixel_mean, dtype=torch.float)[:, None, None]
    batch_cv2im = []
    for ii in range(batch_size):
        rec_im = im_batch[ii].detach().cpu() + pixel_means
        cv2im = rec_im.permute(1, 2, 0).numpy()
        for (x1, y1, x2, y2), is_person, pid in zip(target[ii]['boxes'], target[ii]['labels'], target[ii]['pids'], ):
            if is_person != 1:
                continue
            ec = (255, 0, 0) if pid == -1 else (0, 255, 0)
            if pid != -1:
                # thickness=negative means filled rectangle
                cv2im = cv2.rectangle(cv2im, (x1, y1 - 16), (x2, y1), ec, thickness=-1)
                cv2im = cv2.putText(cv2im, str(int(pid)), org=(x1 + 4, y1 - 2), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            cv2im = cv2.rectangle(cv2im, (x1, y1), (x2, y2), ec, thickness=1)
        # print query pid
        if query_pid is not None:
            cv2im = cv2.putText(cv2im, str(int(query_pid[ii])), org=(25, 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1., color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        batch_cv2im.append(cv2im)

    batch_tensor_im = [torch.as_tensor(im.get()).permute(2, 0, 1) for im in batch_cv2im]
    ret = torch.stack(batch_tensor_im, dim=0)

    # batching images with diff size
    # ret = batching_images_by_padding(batch_tensor_im)
    return ret


def vis_roi_boxes(im_batch, roi_lbl_pid, pixel_means):
    ''' Visualize a mini-batch for debugging.

    Args:
        im_batch (list of 3D tensor): (3 H W) BGR
        roi_lbl_pid (list of 2D tensor): (bs MAX_NUM_ROI_BOXES 6) consist of reg, detect_prob, pid_pred

    Returns:

    '''
    batch_size = len(im_batch)
    pixel_means = torch.tensor(pixel_means, dtype=torch.float)[:, None, None]
    batch_cv2im = []
    for ii in range(batch_size):
        rec_im = im_batch[ii].detach().cpu() + pixel_means  # CHW
        cv2im = rec_im.permute(1, 2, 0).numpy()  # HWC
        for x1, y1, x2, y2, is_person, pid in roi_lbl_pid[ii]:
            if pid > -1:  # id-labeled
                ec = (0, 255, 0)  # (green)
            elif pid == -1 and is_person == 1:  # is-human-wo-id
                ec = (255, 0, 0)  # (blue)
            else:
                ec = (0, 0, 255)  # (red)
            if pid > -1 or is_person == 1:
                # thickness=negative means filled rectangle
                cv2im = cv2.rectangle(cv2im, (x1, y1 - 16), (x2, y1), ec, thickness=-1)
                # cv2im = cv2.rectangle(cv2im, (x1, y2), (x2, y2 + 16), ec, thickness=-1)  # thickness=negative means filled rectangle
                cv2im = cv2.putText(cv2im, "{}&{}".format(str(int(pid)), str(int(is_person))), org=(x1 + 4, y1 - 2),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255),
                                    thickness=1, lineType=cv2.LINE_AA)
            cv2im = cv2.rectangle(cv2im, (x1, y1), (x2, y2), ec, thickness=1)

        batch_cv2im.append(cv2im)
    # .get() convert cv2.UMat to numpy array
    batch_tensor_im = [torch.as_tensor(im.get()).permute(2, 0, 1) for im in batch_cv2im]  # CHW
    ret = torch.stack(batch_tensor_im, dim=0)

    # batching images with diff size
    # ret = batching_images_by_padding(batch_tensor_im)
    return ret


def write_img_with_boxes(img, roi_and_score, pixel_means, file_name=None):
    ''' Visualize a mini-batch for debugging.

    Args:
        img (3D tensor): (3 H W) BGR
        roi_and_score (2D tensor): (MAX_NUM_ROI_BOXES 5) consist of reg, detect_prob

    Returns:

    '''
    assert img.dim() == 3
    pixel_means = torch.tensor(pixel_means, dtype=torch.float)[:, None, None]
    rec_im = img.detach().cpu() + pixel_means
    cv2im = rec_im.permute(1, 2, 0).numpy()
    for x1, y1, x2, y2, score in roi_and_score:
        if score < 0.05:
            continue

        ec = (255, 0, 0)  # (blue)
        cv2im = cv2.rectangle(cv2im, (x1, y1 - 16), (x2, y1), ec,
                              thickness=-1)  # thickness=negative means filled rectangle

        cv2im = cv2.putText(cv2im, "{:.2f}".format(score), org=(x1 + 4, y1 - 2),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255),
                            thickness=1, lineType=cv2.LINE_AA)
        cv2im = cv2.rectangle(cv2im, (x1, y1), (x2, y2), ec, thickness=1)

    # .get() convert cv2.UMat to numpy array
    # cv2im = torch.as_tensor(cv2im.get()).permute(2, 0, 1)
    cv2.imwrite('./cache/demo/result/' + file_name, cv2im)
    return cv2im


def format_detections(num_images, pre_format_dict):
    """Dict{List} to List[Dict]"""
    detections = []
    keys = pre_format_dict.keys()
    for v in pre_format_dict.values(): assert len(v) == num_images

    for i in range(num_images):
        item = dict()
        for k in keys:
            item.update({k: pre_format_dict[k][i]})
        detections.append(item)
    return detections


"""#########
### Test ###
#########"""


def classifier_accuracy(output, target, topk=(1,)):
    ''' classification accuracy

    Args:
        output (tensor): (b n_class)
        target (tensor): (b, )
        topk:

    Returns:

    '''
    # output, target = to_torch(output), to_torch(target)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    ret = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
        ret.append(correct_k.mul_(1. / batch_size))
    return ret[0]


"""#########
### Misc ###
#########"""


class Timer(object):
    """A simple timer."""

    # --------------------------------------------------------
    # Fast R-CNN
    # Copyright (c) 2015 Microsoft
    # Licensed under The MIT License [see LICENSE for details]
    # Written by Ross Girshick
    # --------------------------------------------------------
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def duration(self, diff, average=True):
        self.total_time += diff
        self.calls += 1
        self.average_time = self.total_time / self.calls

        if average:
            return self.average_time
        else:
            return self.diff


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = {}
        self.avg = {}
        self.sum = {}
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        if 0 == self.count:
            self.sum = {k: 0 for k in self.val.keys()}
        self.sum = {curr_k: v + curr_v * n for (k, v), (curr_k, curr_v) in zip(self.sum.items(), val.items())}
        self.count += n
        self.avg = {k: v / self.count for k, v in self.sum.items()}


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def euclidean_dist(x, y):
    """
    Args:
      x: torch tensor, with shape [m, d]
      y: torch tensor, with shape [n, d]
    Returns:
      dist: torch tensor, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def check_scene_boxes_validity(boxes, scene_size, size_thres=1, iou_thres=0.):
    """
    Args:
        box (2D tensor): (num_boxes 4) (x1, y1, x2, y2) x->W y->H
        scene_size (tuple or list): (H W)
    """
    max_h, max_w = scene_size
    inside_boundary = (boxes[:, 0] < (max_w - 1)) & \
                      (boxes[:, 1] < (max_h - 1)) & \
                      (boxes[:, 2] < (max_w + 1)) & \
                      (boxes[:, 3] < (max_h + 1)) & \
                      ((boxes[:, 2] - boxes[:, 0]) > size_thres) & \
                      ((boxes[:, 3] - boxes[:, 1]) > size_thres)
    # should greater than 0
    pos_check = (boxes >= 0).all(dim=1)
    inside_boundary = inside_boundary & pos_check

    # remove lower scoring boxes have an IoU greater than iou_thres with another (higher scoring) box.
    score = ops_box.box_area(boxes)
    nms_sorted = ops_box.nms(boxes, score, iou_thres)
    nms_keep = torch.zeros(boxes.size(0)).type(torch.BoolTensor).to(boxes.device)
    nms_keep[nms_sorted] = True

    valid_index = inside_boundary & nms_keep
    # if torch.sum(valid_index).item() == 0:
    #     return inside_boundary
    return valid_index


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir(os.path.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def read_pil_convert_tensor(path):
    image = imread(path)  # PIL image, RGB
    image = image[:, :, ::-1]  # RGB to BGR
    image = image.astype(np.float32, copy=False)
    image = torch.from_numpy(image).permute(2, 0, 1)  # HWC to CHW
    return image


def split_list_to_chunks(l, chunk_size):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), chunk_size):
        yield l[i:i + chunk_size]


def may_make_dir(path):
    """
    Args:
        path: a dir, e.g. result of `osp.dirname()`
    Note:
        `osp.exists('')` returns `False`, while `osp.exists('.')` returns `True`!
    """
    # This clause has mistakes:
    # if path is None or '':

    if path in [None, '']:
        return
    if not os.path.exists(path):
        os.makedirs(path)


def save_im(im, save_path, transpose=False, check_bound=False):
    """
    im: (1) shape [3, H, W], transpose should be True
        (2) shape [H, W, 3], transpose should be False
        (3) shape [H, W], transpose should be False
    """
    may_make_dir(ospdn(save_path))
    if transpose:
        im = im.transpose(1, 2, 0)
    if check_bound:
        im = im.clip(0, 255)
    im = im.astype(np.uint8)
    mode = 'L' if len(im.shape) == 2 else 'RGB'
    im = Image.fromarray(im, mode=mode)
    im.save(save_path)


def get_model_complexity(model, device):
    model.eval()
    dsize = (1, 3, 600, 1000)
    inputs = [torch.randn(dsize).to(device)]
    targets = [{"boxes": torch.tensor([[200, 50, 300, 250],
                                       [700, 50, 800, 250],
                                       [200, 350, 300, 550],
                                       [700, 350, 800, 550],
                                       [450, 200, 550, 400]
                                       ], dtype=torch.float, device=device)
                }]
    with torch.no_grad():
        total_ops, total_params = profile(model, (inputs, targets, device, True), verbose=False)
    return total_ops / 1e9, total_params / 1e6


@torch.jit.script
def encode_boxes(reference_boxes, proposals, weights):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    """
    Encode a set of proposals with respect to some
    reference boxes

    Arguments:
        reference_boxes (Tensor): reference boxes
        proposals (Tensor): boxes to be encoded
    """

    # perform some unpacking to make it JIT-fusion friendly
    wx = weights[0]
    wy = weights[1]
    ww = weights[2]
    wh = weights[3]

    proposals_x1 = proposals[:, 0].unsqueeze(1)
    proposals_y1 = proposals[:, 1].unsqueeze(1)
    proposals_x2 = proposals[:, 2].unsqueeze(1)
    proposals_y2 = proposals[:, 3].unsqueeze(1)

    reference_boxes_x1 = reference_boxes[:, 0].unsqueeze(1)
    reference_boxes_y1 = reference_boxes[:, 1].unsqueeze(1)
    reference_boxes_x2 = reference_boxes[:, 2].unsqueeze(1)
    reference_boxes_y2 = reference_boxes[:, 3].unsqueeze(1)

    # implementation starts here
    ex_widths = proposals_x2 - proposals_x1
    ex_heights = proposals_y2 - proposals_y1
    ex_ctr_x = proposals_x1 + 0.5 * ex_widths
    ex_ctr_y = proposals_y1 + 0.5 * ex_heights

    gt_widths = reference_boxes_x2 - reference_boxes_x1
    gt_heights = reference_boxes_y2 - reference_boxes_y1
    gt_ctr_x = reference_boxes_x1 + 0.5 * gt_widths
    gt_ctr_y = reference_boxes_y1 + 0.5 * gt_heights

    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * torch.log(gt_widths / ex_widths)
    targets_dh = wh * torch.log(gt_heights / ex_heights)

    targets = torch.cat((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


class BoxCoder(object):
    """
    This class encodes and decodes a set of bounding boxes into
    the representation used for training the regressors.
    """

    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        """
        Arguments:
            weights (4-element tuple)
            bbox_xform_clip (float)
        """
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        boxes_per_image = [len(b) for b in reference_boxes]
        reference_boxes = torch.cat(reference_boxes, dim=0)
        proposals = torch.cat(proposals, dim=0)
        targets = self.encode_single(reference_boxes, proposals)
        return targets.split(boxes_per_image, 0)

    def encode_single(self, reference_boxes, proposals):
        """
        Encode a set of proposals with respect to some
        reference boxes

        Arguments:
            reference_boxes (Tensor): reference boxes
            proposals (Tensor): boxes to be encoded
        """
        dtype = reference_boxes.dtype
        device = reference_boxes.device
        weights = torch.as_tensor(self.weights, dtype=dtype, device=device)
        targets = encode_boxes(reference_boxes, proposals, weights)

        return targets

    def decode(self, rel_codes, boxes):
        assert isinstance(boxes, (list, tuple))
        if isinstance(rel_codes, (list, tuple)):
            rel_codes = torch.cat(rel_codes, dim=0)
        assert isinstance(rel_codes, torch.Tensor)
        boxes_per_image = [len(b) for b in boxes]
        concat_boxes = torch.cat(boxes, dim=0)
        pred_boxes = self.decode_single(
            rel_codes.reshape(sum(boxes_per_image), -1), concat_boxes
        )
        return pred_boxes.reshape(sum(boxes_per_image), -1, 4)

    def decode_single(self, rel_codes, boxes):
        """
        From a set of original boxes and encoded relative box offsets,
        get the decoded boxes.

        Arguments:
            rel_codes (Tensor): encoded boxes
            boxes (Tensor): reference boxes.
        """

        boxes = boxes.to(rel_codes.dtype)

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        dx = rel_codes[:, 0::4] / wx
        dy = rel_codes[:, 1::4] / wy
        dw = rel_codes[:, 2::4] / ww
        dh = rel_codes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(rel_codes)
        # x1
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
        # x2
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
        # y2
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
                matched_vals < self.high_threshold
        )
        matches[below_low_threshold] = Matcher.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = Matcher.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find highest quality match available, even if it is low, including ties
        gt_pred_pairs_of_highest_quality = torch.nonzero(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        """
        Arguments:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentace of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs):
        """
        Arguments:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
            negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.uint8
            )
            pos_idx_per_image_mask[pos_idx_per_image] = 1
            neg_idx_per_image_mask[neg_idx_per_image] = 1

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        return pos_idx, neg_idx
