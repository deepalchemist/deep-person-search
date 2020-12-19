from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops

from lib.misc.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from lib.misc.transform import GeneralizedRCNNTransform
from lib.loss.getloss import CriterionReID
from lib.head.basehead import BaseRoIHeads
from lib.network.getnet import _split_backbone
from lib.head.gethead import FastRCNNPredictor


class OIM(nn.Module):
    def __init__(self, num_classes,
                 # re-ID
                 num_train_pids, cls_type="", cat_c4=False,
                 # Transform parameters
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN parameters
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 box_score_thresh=0.05,
                 box_nms_thresh=0.5, box_detections_per_img=100,
                 # box training
                 box_fg_iou_thresh=0.5,
                 box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 # Misc
                 eval_gt=False, display=False, cws=False,
                 ):

        super(OIM, self).__init__()

        # ------- Backbone -------
        stem, top = _split_backbone('resnet50', load_bgr=True)
        top.representation_size = 2048
        self.backbone = stem

        # ------- RPN -------
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        rpn_kwargs = [
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh
        ]

        rpn_anchor_generator = AnchorGenerator(
            sizes=((8, 16, 32),),
            aspect_ratios=((1, 2),))

        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            RPNHead(in_channels=1024,
                    num_anchors=rpn_anchor_generator.num_anchors_per_location()[0]),
            *rpn_kwargs
        )

        # ------- Box -------
        self.roi_align = MultiScaleRoIAlign(
            featmap_names=["C4"],
            output_size=(14, 7),
            sampling_ratio=0
        )
        representation_size = top.representation_size
        box_predictor = FastRCNNPredictor(representation_size, num_classes)

        box_kwargs = [
            # Faster R-CNN training
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            # Faster R-CNN inference
            box_score_thresh, box_nms_thresh, box_detections_per_img
        ]
        embedding_head = ExtractReIDFeat(
            featmap_names=['C4', 'C5'] if cat_c4 else ['C5'],
            in_channels=[1024, 2048] if cat_c4 else [2048],
            dim=256
        )
        reid_loss = CriterionReID(
            cls_type,
            256,
            num_train_pids
        )
        feat_head = RCNNConvHead(top)

        self.roi_heads = RoIHeads(
            embedding_head, reid_loss,
            self.roi_align, feat_head, box_predictor,
            *box_kwargs
        )
        self.roi_heads.cws = cws

        self.req_pid = -1 if cls_type == "oim" else 0
        self.reid_time = 0

        # ------- Misc -------
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]  # NOTE: RGB order is given here
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
        self.eval_gt = eval_gt
        self.display = display

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.backbone.train(mode)
        self.roi_heads.train(mode)
        return self

    def extra_box_feat(self, images, targets):
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("C4", features)])
        proposals = [targets[0]['boxes']]  # should be (1 4)

        roi_pooled_features = self.roi_heads.box_roi_pool(
            features, proposals, images.image_sizes)

        rcnn_features = self.roi_heads.feat_head(roi_pooled_features)
        if isinstance(rcnn_features, torch.Tensor):
            rcnn_features = OrderedDict([('C5', rcnn_features)])
        embeddings = self.roi_heads.embedding_head(rcnn_features)
        return embeddings

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)  # Cov4 4D tensor

        if isinstance(features, torch.Tensor):
            features = OrderedDict([("C4", features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return detections, losses


class RCNNConvHead(nn.Sequential):

    def __init__(self, model):
        super(RCNNConvHead, self).__init__(
            OrderedDict(
                [('top', model)]  #
            )
        )
        self.out_channels = [1024, 2048]

    def forward(self, x):
        feat = super(RCNNConvHead, self).forward(x)
        return {'C4': F.adaptive_max_pool2d(x, 1),
                'C5': F.adaptive_max_pool2d(feat, 1)}  # Global Max Pooling


class RoIHeads(BaseRoIHeads):

    def __init__(self, embedding_head, reid_loss, *args, **kwargs):
        super(RoIHeads, self).__init__(*args, **kwargs)
        self.embedding_head = embedding_head
        self.reid_loss = reid_loss
        self.cws = False

    @property
    def feat_head(self):
        return self.box_head

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                assert t["boxes"].dtype.is_floating_point, \
                    'target boxes must of float type'
                assert t["labels"].dtype == torch.int32, \
                    'target labels must of int32 type'
                assert t["pids"].dtype == torch.int64, \
                    'target pids must of int64 type'
        result, losses = {},{}
        if self.training:
            proposals, matched_idxs, labels, regression_targets, pids = \
                self.select_training_samples(proposals, targets)
            
            result.update({"boxes": proposals, "pids":pids})

        roi_pooled_features = \
            self.box_roi_pool(features, proposals, image_shapes)
        rcnn_features = self.feat_head(roi_pooled_features)  # max_pooled c4 and c5 features
        class_logits, box_regression = self.box_predictor(rcnn_features['C5'])
        embeddings_ = self.embedding_head(rcnn_features)
        
        if self.training:
            loss_detection, loss_box_reg = \
                fastrcnn_loss(class_logits, box_regression,
                              labels, regression_targets)

            loss_reid, acc = self.reid_loss(embeddings_, pids)

            losses = dict(loss_classifier=loss_detection,
                          loss_box_reg=loss_box_reg,
                          loss_ide=loss_reid)
            
            result.update({"acc": acc})
        else:
            boxes, scores, embeddings, labels = \
                self._postprocess_detections(class_logits, box_regression, embeddings_,
                                             proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i],
                        labels=labels[i],
                        scores=scores[i],
                        feats=embeddings[i],
                    )
                )
        return result, losses

    def _postprocess_detections(self,
                                class_logits,
                                box_regression,
                                embeddings_,
                                proposals,
                                image_shapes):
        device = class_logits.device

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)[:, 1:]  # fist class is background

        if self.cws:
            embeddings_ = embeddings_ * pred_scores.view(-1, 1)  # CWS

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        pred_embeddings = embeddings_.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_embeddings = []
        for boxes, scores, embeddings, image_shape in zip(pred_boxes, pred_scores, pred_embeddings, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.ones(scores.size(0), device=device)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores.unsqueeze(1)
            labels = labels.unsqueeze(1)
            # embeddings are already personized.

            # batch everything, by making every class prediction be a separate
            # instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()
            embeddings = embeddings.reshape(-1, self.embedding_head.dim)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels, embeddings = boxes[
                                                    inds], scores[inds], labels[inds], embeddings[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, embeddings = boxes[keep], scores[keep], \
                                                labels[keep], embeddings[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels, embeddings = boxes[keep], scores[keep], \
                                                labels[keep], embeddings[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_embeddings.append(embeddings)

        return all_boxes, all_scores, all_embeddings, all_labels


class ExtractReIDFeat(nn.Module):

    def __init__(self, featmap_names=['C5'],
                 in_channels=[2048],
                 dim=256):
        super(ExtractReIDFeat, self).__init__()
        self.cat_c4 = True if len(featmap_names) > 1 else False
        self.featmap_names = featmap_names
        self.in_channels = list(map(int, in_channels))
        self.dim = int(dim)

        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        indv_dims = list(map(int, indv_dims))
        for ftname, in_chennel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            proj = nn.Sequential(
                nn.Linear(in_chennel, indv_dim, bias=False),
                nn.BatchNorm1d(indv_dim))
            nn.init.normal_(proj[0].weight, std=0.01)
            self.projectors[ftname] = proj

    def forward(self, featmaps):
        '''
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        '''
        assert len(featmaps) == 2
        if not self.cat_c4:
            featmaps.pop("C4")

        if len(featmaps) == 1:
            k, v = list(featmaps.items())[0]
            v = self._flatten_fc_input(v)
            embeddings = self.projectors[k](v)
        else:
            outputs = []
            for k, v in featmaps.items():
                v = self._flatten_fc_input(v)
                outputs.append(
                    self.projectors[k](v)
                )
            embeddings = torch.cat(outputs, dim=1)  # (N, dim)

        if not self.training:
            embeddings = F.normalize(embeddings, dim=1)

        return embeddings

    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x  # ndim = 2, (N, d)

    def _split_embedding_dim(self):
        parts = len(self.in_channels)
        tmp = [self.dim / parts] * parts
        if sum(tmp) == self.dim:
            return tmp
        else:
            res = self.dim % parts
            for i in range(1, res + 1):
                tmp[-i] += 1
            assert sum(tmp) == self.dim
            return tmp


def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
    """
    Computes the loss for Faster R-CNN.

    Arguments:
        class_logits (Tensor): include background
        box_regression (Tensor)

    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """

    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()  # box_weight=1 yield better performances

    return classification_loss, box_loss

