import time
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops
from torchvision.ops import MultiScaleRoIAlign

from lib.misc import util
from lib.misc.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from lib.misc.transform import GeneralizedRCNNTransform
from lib.loss.getloss import CriterionReID
from lib.head.embedding import EmbDet
from lib.head.gethead import FastRCNNPredictor
from lib.head.basehead import BaseRoIHeads
from lib.network.getnet import _split_backbone
from lib.detector.getdet import DetectorBackbone


class BSL(nn.Module):
    def __init__(self, num_classes,
                 # re-ID
                 num_train_pids, cls_type="", in_level=["C5"],
                 # Transform
                 min_size=800, max_size=1333,
                 image_mean=None, image_std=None,
                 # RPN
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box
                 box_score_thresh=0.05, box_nms_thresh=0.5, box_detections_per_img=100,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_batch_size_per_image=512, box_positive_fraction=0.25,
                 bbox_reg_weights=None,
                 # Misc
                 eval_gt=False, display=False, cws=False
                 ):

        super(BSL, self).__init__()
        # ------- Backbone -------
        base_model, top_model = _split_backbone(backbone_name='resnet50', conv5_stride=2)
        return_layers = {
            'conv1': "C1",
            'conv2': "C2",
            'conv3': "C3",
            'conv4_3': "C4",
        }
        self.backbone = DetectorBackbone(base_model, return_layers)

        # ------- RPN -------
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        rpn_kwargs = [rpn_fg_iou_thresh, rpn_bg_iou_thresh,
                      rpn_batch_size_per_image, rpn_positive_fraction,
                      rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh]
        rpn_anchor_generator = AnchorGenerator(
            sizes=((8, 16, 32),),
            aspect_ratios=((1, 2),))
        self.RPN = RegionProposalNetwork(
            rpn_anchor_generator,
            RPNHead(1024, rpn_anchor_generator.num_anchors_per_location()[0]),
            *rpn_kwargs
        )

        # ------- R-CNN -------
        roi_align = MultiScaleRoIAlign(
            featmap_names=["C4"],
            output_size=(14, 7),
            sampling_ratio=0
        )
        resolution_h, resolution_w = roi_align.output_size[0], roi_align.output_size[1]

        box_emb = EmbDet(1024, 256, resolutions=[resolution_h, resolution_w])
        box_predictor = FastRCNNPredictor(box_emb.representation_size, num_classes)

        box_kwargs = [
            # Faster R-CNN training
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            # Faster R-CNN inference
            box_score_thresh, box_nms_thresh, box_detections_per_img
        ]
        self.RCNN = RCNN(
            roi_align, box_emb, box_predictor,
            *box_kwargs
        )
        self.RCNN.cws = cws

        # ------- re-ID -------
        out_channels = 256
        in_ch_list = [2048, 1024, 512, 256, 256][:len(in_level)][::-1]
        reid_emb = EmbedReID(
            top_model,
            roi_align,
            featmap_names=in_level,
            in_ch_list=in_ch_list,
            out_ch=out_channels
        )
        reid_crit = nn.ModuleDict()
        for name, in_ch in zip(in_level, in_ch_list):
            reid_crit[name] = CriterionReID(cls_type, in_ch, num_train_pids)

        self.reid_head = ReIDHead(
            reid_emb,
            reid_crit,
            # PK sampling
            n_roi_per_gt=4,
            fg_iou_thresh=0.5
        )

        # -------- Others -------
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
        return self

    def extra_box_feat(self, images, targets):
        """
        images (list[Tensor]): length=1
        targets (list[Dict[Tensor]]): length=1
        """
        assert len(images) == len(targets) == 1, "Only support single image input"

        images, targets = self.transform(images, targets)
        # Backbone
        x = self.backbone(images.tensors)
        x = x['C4']

        box_coord = [targets[0]['boxes']]  # should be (1 4)
        # box features
        results, _ = self.reid_head(OrderedDict([("C4", x)]), box_coord, images.image_sizes)
        box_feat = results['feats']

        return box_feat

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should not be None.")

        # ---------------------------------------------------------------------------------------------------
        # Data pre-processing
        num_images = len(images)
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)

        # Backbone forward
        featmaps = self.backbone(images.tensors)
        x = featmaps['C4']
        c4_det = featmaps['C4']
        c4_reid = featmaps['C4']

        # ---------------------------------------------------------------------------------------------------
        # RPN
        # List[Tensor(post_nms_top_n 4)], Dict{losses}, len(proposals)=batch_size
        proposals, proposal_losses = self.RPN(images, OrderedDict([("C4", x)]), targets)
        if not self.training and self.eval_gt:
            proposals = [t['boxes'] for t in targets]

        # R-CNN
        # Dict{List} "fg_cnt", "bg_cnt" in training, Dict{Tensor} 'class_logits','box_regressions' in test, Dict{losses}
        det_res, rcnn_losses = self.RCNN(OrderedDict([("C4", c4_det)]), proposals, images.image_sizes, targets)

        # ---------------------------------------------------------------------------------------------------
        # re-ID Head
        # pooling re-ID RoI feature using R-CNN detections
        # reid_props = self.RCNN.box_decoder(detections['box_regression'], proposals, images.image_sizes)
        reid_tic = time.time()
        # Dict{List} "boxes", "pids", "acc" in training, Dict{Tensor} "feats" in test. Dict{losses}
        reid_res, reid_losses = self.reid_head(
            OrderedDict([("C4", c4_reid)]), proposals, images.image_sizes, targets
        )
        self.reid_time = time.time() - reid_tic

        # ---------------------------------------------------------------------------------------------------
        # Collecting detections
        detections = self.collect_detections(reid_res, det_res, num_images, images, proposals, targets)
        # mapping boxes to origin image size, only return input when training
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        # collect losses
        losses = {}
        losses.update(proposal_losses)
        losses.update(rcnn_losses)
        losses.update(reid_losses)

        # Return
        return detections, losses

    def collect_detections(self, reid_res, det_res, num_images, images, proposals, targets):
        detections = []
        if self.training:
            reid_res.update(det_res)  # "boxes", "pids", "acc", "fg_cnt", "bg_cnt"
            if self.display:
                reid_res.update({"img": [img.detach() for img in images.tensors], "tgt": targets})
                reid_res.update({"labels": [torch.ones(t.size(0), dtype=torch.long, device=t.device)
                                            for t in reid_res['boxes']]})
            detections = util.format_detections(num_images, reid_res)  # Dict{List} to List[Dict]
        else:
            class_logits, box_regression, box_feats = \
                det_res['class_logits'], det_res['box_regression'], reid_res['feats']
            assert class_logits.size(0) == box_regression.size(0) == box_feats.size(0)
            # boxes: List[Tensor(detections_per_img (num_cls-1)*4)]
            if self.eval_gt:
                boxes = [t['boxes'] for t in targets]
                scores = [torch.ones(b.size(0)).to(b.device) for b in boxes]
                labels = [torch.ones(b.size(0), dtype=torch.long).to(b.device) for b in boxes]
                box_feats = box_feats.split([b.size(0) for b in boxes], dim=0)
            else:
                boxes, scores, labels, box_feats = self.RCNN._postprocess_detections(
                    class_logits, box_regression, proposals, images.image_sizes, box_feats, mode="rcnn")

            # One image one Dict
            for i in range(num_images):
                detections.append(
                    dict(
                        boxes=boxes[i],  # box coordinates
                        labels=labels[i],  # class index, e.g., bg or person
                        scores=scores[i],  # classification confidence
                        feats=box_feats[i]  # reid features of boxes
                    )
                )
        return detections


class RCNN(BaseRoIHeads):
    def __init__(self,
                 box_roi_pool,
                 box_head,
                 box_predictor,
                 # Faster R-CNN training
                 fg_iou_thresh, bg_iou_thresh,
                 batch_size_per_image, positive_fraction,
                 bbox_reg_weights,
                 # Faster R-CNN inference
                 score_thresh,
                 nms_thresh,
                 detections_per_img,
                 ):
        super(RCNN, self).__init__(box_roi_pool,
                                   box_head,
                                   box_predictor,
                                   # Faster R-CNN training
                                   fg_iou_thresh, bg_iou_thresh,
                                   batch_size_per_image, positive_fraction,
                                   bbox_reg_weights,
                                   # Faster R-CNN inference
                                   score_thresh,
                                   nms_thresh,
                                   detections_per_img, )
        self.cws = False
    def forward(self,
                features,
                proposals,
                image_shapes,
                targets=None):
        """
        Arguments:
            features (Dict[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
        """
        if self.training:
            proposals, matched_idxs, labels, regression_targets, _ = \
                self.select_training_samples(proposals, targets)

        box_features = self.box_roi_pool(features, proposals, image_shapes)  # (n_roi_per_img*bs c h w)
        box_features = self.box_head(box_features)  # (n_roi_per_img*bs dim_feat)
        class_logits, box_regression = \
            self.box_predictor(box_features)  # (n_roi_per_img*bs n_cls) (n_roi_per_img*bs 4)

        result, losses = {}, {}
        if self.training:
            loss_classifier, loss_box_reg = _fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)
            # return for re-ID and visualization
            fg_cnt = [torch.sum(label == 1).item() for label in labels]
            bg_cnt = [torch.sum(label == 0).item() for label in labels]
            result.update({"fg_cnt": fg_cnt, "bg_cnt": bg_cnt})
        else:
            result.update({"class_logits": class_logits, "box_regression": box_regression})
        return result, losses

    def _postprocess_detections(self,
                                class_logits,
                                box_regression,
                                proposals,
                                image_shapes,
                                box_features,
                                mode="rcnn"):
        """
        class_logits: 2D tensor(n_roi_per_img*bs C)
        box_regression: 2D tensor(n_roi_per_img*bs C*4)
        proposals: list[tensor(n_roi_per_img 4)]
        image_shapes: list[tuple[H, W]]
        box_features: 2D tensor(n_roi_per_img*bs dim_feat)]
        mode: test with RPN or RCNN detections
        """

        device = class_logits.device
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)  # tensor(n_roi_per_img*bs C 4)

        pred_scores = F.softmax(class_logits, -1)
        pred_scores = pred_scores[:, 1:]

        if self.cws:
            box_features = box_features * pred_scores.view(-1, 1)  # CWS
        box_features = box_features.split(boxes_per_image, 0)

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)  # list[tensor(n_roi_per_img C 4)], length=bs
        pred_scores = pred_scores.split(boxes_per_image, 0)  # list[tensor(n_roi_per_img 1)], length=bs

        all_boxes = []
        all_scores = []
        all_labels = []
        all_feats = []
        n_iter = 0
        # go through batch_size
        for boxes, scores, image_shape in zip(pred_boxes, pred_scores, image_shapes):
            #
            if box_features is not None:
                features = box_features[n_iter]

            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)  # tensor(n_roi_per_img C 4)

            # create labels for each prediction
            labels = torch.ones(scores.size(0), device=device)

            # remove predictions with the background label
            boxes = boxes[:, 1:]  # tensor(n_roi_per_img C-1 4)
            labels = labels.unsqueeze(1)  # tensor(n_roi_per_img 1)

            ### using rpn proposals for testing ###
            if "rpn" == mode:
                boxes = proposals[n_iter]
            #######################################

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)  # 2D tensor(n_roi_per_img*(C-1) 4)
            scores = scores.flatten()
            labels = labels.flatten()

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]
            if box_features is not None:
                features = features[inds]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            if box_features is not None:
                features = features[keep]
            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            if box_features is not None:
                all_feats.append(features)
            n_iter += 1
        return all_boxes, all_scores, all_labels, all_feats


def _fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
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
    # classification_loss = F.binary_cross_entropy_with_logits(class_logits.squeeze(1), labels.float())

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
    box_loss = 1. * box_loss / labels.numel()

    return classification_loss, box_loss


class EmbedReID(nn.Module):
    def __init__(self, top_model,
                 roi_align,
                 featmap_names=["C5"],
                 in_ch_list=[2048],
                 out_ch=256,
                 ):
        super(EmbedReID, self).__init__()
        assert "C5" in featmap_names

        # roi pooling
        self.roi_align = roi_align
        # projectors
        self.projectors = nn.ModuleDict()
        for name, in_ch in zip(featmap_names, in_ch_list):
            self.projectors[name] = nn.BatchNorm1d(in_ch)

        self.top = top_model

    def forward(self, feature, proposals, image_shapes):
        c4_roi_featmap = self.roi_align(feature, proposals, image_shapes)  # (n_roi_per_img*bs c h w)
        c5_roi_featmap = self.top(c4_roi_featmap)
        inputs = {
            "C4": F.adaptive_max_pool2d(c4_roi_featmap, 1).flatten(start_dim=1),
            "C5": F.adaptive_max_pool2d(c5_roi_featmap, 1).flatten(start_dim=1)
        }
        output = {}
        for name, projector in self.projectors.items():
            output.update({
                name: projector(inputs[name])
            })
        return output


class ReIDHead(torch.nn.Module):
    def __init__(self,
                 extractor,
                 criterion,
                 # re-ID training
                 n_roi_per_gt,
                 fg_iou_thresh,
                 ):
        super(ReIDHead, self).__init__()
        self.box_similarity = box_ops.box_iou
        self.n_roi_per_gt = n_roi_per_gt
        self.fg_iou_thresh = fg_iou_thresh

        self.extractor = extractor
        self.criterion = criterion
        self.req_pid = -1 if criterion['C5'].cls_type == "oim" else 0

    def add_gt_proposals(self, proposals, gt_boxes):
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]
        return proposals

    def check_targets(self, targets):
        assert targets is not None
        assert all("boxes" in t for t in targets)
        assert all("pids" in t for t in targets)

    def _select_training_samples(self, proposals, targets):
        """
        proposals: list[tensor(post_nms_top_n 4)]
        targets: list[dict]
        """
        self.check_targets(targets)
        gt_boxes = [t["boxes"] for t in targets]
        gt_pids = [t["pids"] for t in targets]
        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)  # list[tensor(post_nms_top_n+n_gt_boxes 4)]

        matched_props = []
        matched_pids = []
        match_roi_idxes = []  # indexes of the chosen rois
        # match_gt_idxes = []  # gt index of the chosen rois

        # go through batch images
        for proposals_in_image, gt_boxes_in_image, gt_pids_in_image in zip(proposals, gt_boxes, gt_pids):
            # IoU matrix (#gt_boxes, #anchors)
            match_quality_matrix = self.box_similarity(gt_boxes_in_image, proposals_in_image)
            # todo-note: make sure each propo assigned to single gt
            max_values, max_indexes = match_quality_matrix.max(dim=0)
            # O-1 matrix (#gt_boxes, #anchors)
            matches = match_quality_matrix == max_values[None, :]
            # IoU threshold
            thresholded = match_quality_matrix >= self.fg_iou_thresh
            # 0-1 matrix (#gt_boxes, #anchors)
            matches = (matches & thresholded).float()
            thr_match_quality_matrix = match_quality_matrix * matches  # (#gt_boxes, #anchors), thresholded(max-value) or 0
            # filter min(top-k, num_valid) per gt_boxes
            val, idx = thr_match_quality_matrix.topk(self.n_roi_per_gt, dim=1)  # (#gt_boxes, n_roi_per_gt)
            sampled_inds = idx[val > 0]  # indexes of the chosen rois
            match_roi_idxes.append(sampled_inds)
            matched_props.append(proposals_in_image[sampled_inds])
            matched_pids.append(gt_pids_in_image[max_indexes][sampled_inds])

            # gt_index = torch.tensor(range(val.size(0)), device=val.device)[:, None].expand_as(val)
            # match_gt_idxes.append(gt_index[val > 0])

        return matched_props, matched_pids, match_roi_idxes

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        features (OrderDict)
        proposals (List[tensor(post_nms_top_n 4)])
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

        num_images = len(proposals)
        result, reid_loss = {}, {}

        if self.training:
            # PK sampling positive proposals for re-ID training
            proposals, pids, match_gt_idx = self._select_training_samples(proposals, targets)
            if self.req_pid == 0:
                keeps, pids, proposals = util.filter_required_pids(pids, proposals, self.req_pid)
                # match_gt_idx = [m[k] for m, k in zip(match_gt_idx, keeps)]

            result.update({"boxes": proposals, "pids": pids})

        # num_roi_per_img = [p.size(0) for p in proposals]
        # Dict{"":Tensor(n_roi_per_img*bs dim_feat)}
        box_features = self.extractor(features, proposals, image_shapes)

        if self.training:
            # dict, top-1 acc
            loss_ide_list, acc_list = [], []
            for name, feat in box_features.items():
                loss_ide, acc = self.criterion[name](feat, pids=pids)
                loss_ide_list.append(loss_ide)
                acc_list.append(acc)

            reid_loss.update({"loss_ide": sum(loss_ide_list) / len(loss_ide_list)})
            result.update({"acc": [sum(acc) / len(acc)] * num_images})

        # box_features = box_features.split(num_roi_per_img, dim=0)
        # assert len(box_features) == num_images
        if not self.training:
            box_features = [F.normalize(feat, dim=1) for feat in box_features.values()]
            box_features = torch.cat(box_features, dim=1)
            result.update({"feats": box_features})

        return result, reid_loss
