import torch
import torch.nn.functional as F

from torchvision.ops import boxes as box_ops
from lib.misc import util

global logger

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
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss


class BaseRoIHeads(torch.nn.Module):
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
        super(BaseRoIHeads, self).__init__()
        self.box_similarity = box_ops.box_iou
        # assign ground-truth boxes for each proposal
        self.proposal_matcher = util.Matcher(
            fg_iou_thresh,
            bg_iou_thresh,
            allow_low_quality_matches=False)

        self.fg_bg_sampler = util.BalancedPositiveNegativeSampler(
            batch_size_per_image,
            positive_fraction)

        if bbox_reg_weights is None:
            bbox_reg_weights = (10., 10., 5., 5.)
        self.box_coder = util.BoxCoder(bbox_reg_weights)

        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_predictor = box_predictor

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def has_mask(self):
        return False

    def has_keypoint(self):
        return False

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels, gt_pids):
        matched_idxs = []
        labels = []
        pids = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image, gt_pids_in_image in \
                zip(proposals, gt_boxes, gt_labels, gt_pids):
            match_quality_matrix = self.box_similarity(gt_boxes_in_image, proposals_in_image)
            matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

            clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

            labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
            labels_in_image = labels_in_image.to(dtype=torch.int64)

            ### added
            pids_in_image = gt_pids_in_image[clamped_matched_idxs_in_image]
            pids_in_image = pids_in_image.to(dtype=torch.int64)
            ###

            # Label background (below the low threshold)
            bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_in_image[bg_inds] = 0
            pids_in_image[bg_inds] = -2  # label -2 for non-human boxes, -1 for is-human-but-non-ID boxes

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler
            pids_in_image[ignore_inds] = -3  # -3 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
            pids.append(pids_in_image)
        return matched_idxs, labels, pids

    def subsample(self, labels):
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
                zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def check_targets(self, targets):
        assert targets is not None
        assert all("boxes" in t for t in targets)
        assert all("labels" in t for t in targets)
        if self.has_mask():
            assert all("masks" in t for t in targets)

    def select_training_samples(self, proposals, targets):
        self.check_targets(targets)
        gt_boxes = [t["boxes"] for t in targets]
        gt_labels = [t["labels"] for t in targets]
        gt_pids = [t["pids"] for t in targets]

        # append ground-truth bboxes to propos
        # TODO(BUG) negative coordinate of gt_boxes raise error
        proposals = self.add_gt_proposals(proposals, gt_boxes)  # size of proposals (2k+n_gt_boxes 4)

        # get matching gt indices for each proposal
        matched_idxs, labels, pids = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels, gt_pids)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)  # list of (n_roi_per_img)
        matched_gt_boxes = []

        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            ### added
            pids[img_id] = pids[img_id][img_sampled_inds]
            ###
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            matched_gt_boxes.append(gt_boxes[img_id][matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets, pids

    def box_decoder(self, box_regression, proposals, image_shapes):
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        all_boxes = []
        for boxes, image_shape in zip(pred_boxes, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            boxes = boxes[:, 1:]
            boxes = boxes.reshape(-1, 4)
            all_boxes.append(boxes)
        return all_boxes

    def postprocess_detections(self,
                               class_logits,
                               box_regression,
                               proposals,
                               image_shapes,
                               box_features=None,
                               mode="rcnn"):
        """
        class_logits: 2D tensor(n_roi_per_img*bs C)
        box_regression: 2D tensor(n_roi_per_img*bs C*4)
        proposals: list[tensor(n_roi_per_img 4)]
        image_shapes: list[tuple[H, W]]
        box_features: 2D tensor(n_roi_per_img*bs dim_feat)]
        """

        device = class_logits.device
        num_classes = class_logits.shape[-1]
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)  # tensor(n_roi_per_img*bs C 4)

        pred_scores = F.softmax(class_logits, -1)
        # pred_scores = torch.sigmoid(class_logits)

        ### added
        if box_features is not None:
            # box_features = box_features * pred_scores.view(-1, 1)  # CWS
            box_features = box_features.split(boxes_per_image, 0)
        ###

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)  # list[tensor(n_roi_per_img C 4)], length=bs
        pred_scores = pred_scores.split(boxes_per_image, 0)  # list[tensor(n_roi_per_img C)], length=bs

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
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]  # tensor(n_roi_per_img C-1 4)
            scores = scores[:, 1:]  # tensor(n_roi_per_img C-1)
            labels = labels[:, 1:]  # tensor(n_roi_per_img C-1)

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

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if self.training:
            proposals, matched_idxs, labels, regression_targets = \
                self.select_training_samples(proposals, targets)
        box_features = self.box_roi_pool(features, proposals, image_shapes)  # (n_roi_per_img*batch_size 1024 14 14)
        box_features = self.box_head(box_features)
        # tensor(n_roi_per_img*bs C), tensor(n_roi_per_img*bs C*4)
        class_logits, box_regression = self.box_predictor(box_features)

        result, losses = [], {}
        if self.training:
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg)
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i],
                        labels=labels[i],
                        scores=scores[i],
                    )
                )
        return result, losses
