import os
import json
import pickle
import os.path as osp
from PIL import Image
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, precision_recall_curve

from lib.data.imdb import imdb


def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union


class ssm(imdb):
    """
    Also known as CUHK-SYSU
    18184 images in total
    Train: #image 11206, 1490 images without id-labeled box
    #Train ID 5532 (not include -1, labeled from 0 to N-1), #boxes 55272,
    Train: #box w/ID 15085 (correspond to ID 0 to N-1), #box wo/ID 40187 (correspond to ID -1)
    Test: #image 6978, #ID 2900, #boxes 40871
    Test: #box w/ID 8345, #box wo/ID 32526
    re-ID dataset #train_boxes=15085(not include 40187 unlabeled), #query_boxes=2900, #gallery_boxes=37971
    """

    def __init__(self, image_set, data_type='origin', probe_type='origin', use_mask=False, root_dir=None):
        super(ssm, self).__init__('ssm_' + image_set)
        self._image_set = image_set
        self._root_dir = os.path.join(root_dir, 'ssm')
        self.data_path = data_type
        assert osp.isdir(self._root_dir), "Path does not exist: {}".format(self._root_dir)
        assert osp.isdir(self._data_path), "Path does not exist: {}".format(self._data_path)

        self._num_train_ids = 5532
        self._id2idx = self._train_id_to_index()

        self._image_names = self._load_split_img_name()
        self.probes = probe_type
        self._roidb = self.fetch_gt_roidb()

        if use_mask:
            # Binary masks of pedestrians, in (h, w), {0, 255}
            mask_dir = osp.join(self._root_dir, 'mask')
            self.set_mask_dir(mask_dir)
            assert osp.isdir(self._mask_dir), "Mask does not exist: {}".format(self._mask_dir)
            self.add_mask_to_roidb()

    @property
    def data_path(self):
        return self._data_path

    @data_path.setter
    def data_path(self, type='origin'):
        data_type = {
            "origin": "Image/SSM",
        }
        self._data_path = osp.join(self._root_dir, data_type[type])

    @property
    def probes(self):
        return self._probes

    @probes.setter
    def probes(self, type='origin'):
        probe_type = {
            "origin": self._load_probes,
        }
        assert type in probe_type
        self._probes = probe_type[type]()

    def _train_id_to_index(self):
        return {ii: ii for ii in range(self.num_train_ids)}

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        image_path = osp.join(self._data_path, self._image_names[i])
        assert osp.isfile(image_path), \
            "Path does not exist: {}".format(image_path)
        return image_path

    def mask_path_at(self, i):
        """
        Return the absolute path to mask of image i in the sequence.
        """
        mask_path = osp.join(self._mask_path, self._image_names[i])
        assert osp.isfile(mask_path), \
            "Mask does not exist: {}".format(mask_path)
        return mask_path

    def _load_split_img_name(self):
        """
        Load the indexes for the specific subset (train / test).
        For PSDB, the index is just the image file name.
        """
        # test pool
        test = loadmat(osp.join(self._root_dir, 'annotation', 'pool.mat'))
        test = test['pool'].squeeze()
        test = [str(a[0]) for a in test]
        if self._image_set == 'test': return test

        # all images
        all_imgs = loadmat(osp.join(self._root_dir, 'annotation', 'Images.mat'))
        all_imgs = all_imgs['Img'].squeeze()
        all_imgs = [str(a[0][0]) for a in all_imgs]
        # training
        return list(set(all_imgs) - set(test))

    def _load_probes(self):
        """
        Load the list of (img, roi) for probes. For test split, it's defined
        by the protocol. For training split, will randomly choose some samples
        from the gallery as probes.
        """
        protoc = loadmat(osp.join(self._root_dir, 'annotation/test/train_test/TestG50.mat'))['TestG50'].squeeze()
        probes = []
        for item in protoc['Query']:
            im_path = osp.join(self._data_path, str(item['imname'][0, 0][0]))
            roi = item['idlocate'][0, 0][0].astype(np.int32)
            roi[2:] += roi[:2]
            # pid = int(item['idname'][0][0][0].split('p')[-1])
            probes.append((im_path, roi))
        return probes

    def fetch_gt_roidb(self):
        # Load all images and build a dict from image to boxes
        all_imgs = loadmat(osp.join(self._root_dir, 'annotation', 'Images.mat'))
        all_imgs = all_imgs['Img'].squeeze()
        name_to_boxes = {}
        name_to_pids = {}
        for im_name, __, boxes in all_imgs:
            im_name = str(im_name[0])
            boxes = np.asarray([b[0] for b in boxes[0]])
            boxes = boxes.reshape(boxes.shape[0], 4)
            valid_index = np.where((boxes[:, 2] > 0) & (boxes[:, 3] > 0))[0]
            assert valid_index.size > 0, \
                'Warning: {} has no valid boxes.'.format(im_name)
            boxes = boxes[valid_index]
            name_to_boxes[im_name] = boxes.astype(np.int32)
            # todo(note): unlabeled background person id = -1
            name_to_pids[im_name] = -1 * np.ones(boxes.shape[0], dtype=np.int32)

        def _set_box_pid(boxes, box, pids, pid):
            for i in range(boxes.shape[0]):
                if np.all(boxes[i] == box):
                    pids[i] = pid
                    return
            print('Warning: person {} box {} cannot find in Images'.format(pid, box))

        # todo(note): Load all the train / test persons and label their pids from 0 to N-1,
        #  Assign pid = -1 for is_person_without_pid
        if self._image_set == 'train':
            train = loadmat(osp.join(self._root_dir, 'annotation/test/train_test/Train.mat'))
            train = train['Train'].squeeze()  # ID to its scenes
            for index, item in enumerate(train):  # index denotes ID
                scenes = item[0, 0][2].squeeze()
                for im_name, box, __ in scenes:
                    im_name = str(im_name[0])
                    box = box.squeeze().astype(np.int32)
                    _set_box_pid(name_to_boxes[im_name], box,
                                 name_to_pids[im_name], index)
        else:
            test = loadmat(osp.join(self._root_dir, 'annotation/test/train_test/TestG50.mat'))
            test = test['TestG50'].squeeze()
            for index, item in enumerate(test):
                # query
                im_name = str(item['Query'][0, 0][0][0])
                box = item['Query'][0, 0][1].squeeze().astype(np.int32)
                _set_box_pid(name_to_boxes[im_name], box,
                             name_to_pids[im_name], index)
                # gallery
                gallery = item['Gallery'].squeeze()
                for im_name, box, __ in gallery:
                    im_name = str(im_name[0])
                    if box.size == 0: break
                    box = box.squeeze().astype(np.int32)
                    _set_box_pid(name_to_boxes[im_name], box,
                                 name_to_pids[im_name], index)

        # Construct the gt_roidb
        gt_roidb = []
        for index, im_name in enumerate(self.image_names):
            img_size = Image.open(self.image_path_at(index)).size  # (W H)
            boxes = name_to_boxes[im_name]
            boxes[:, 2] += boxes[:, 0]  # W
            boxes[:, 3] += boxes[:, 1]  # H
            pids = name_to_pids[im_name].astype(np.long)
            num_objs = len(boxes)
            gt_classes = np.ones((num_objs), dtype=np.int32)

            gt_roidb.append({
                # scene image
                'img_name': im_name,
                'img_path': self.image_path_at(index),
                'width': img_size[0],
                'height': img_size[1],
                'flipped': False,
                # box
                'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_pids': pids,
            })

        return gt_roidb

    def evaluate_detections(self, gallery_det, score_thresh=0.5, iou_thresh=0.5,
                            labeled_only=False):
        """
        gallery_det (list of ndarray): n_det x [x1, x2, y1, y2, score] per image

        det_thresh (float): filter out gallery detections whose scores below this
        iou_thresh (float): treat as true positive if IoU is above this threshold
        labeled_only (bool): filter out unlabeled background people
        """
        assert self.num_images == len(gallery_det)

        gt_roidb = self.fetch_gt_roidb()
        y_true, y_score = [], []
        count_gt, count_tp = 0, 0
        for gt, det in zip(gt_roidb, gallery_det):
            gt_boxes = gt['boxes']
            if labeled_only:
                inds = np.where(gt['gt_pids'].ravel() > 0)[0]
                if len(inds) == 0: continue
                gt_boxes = gt_boxes[inds]
            det = np.asarray(det)
            inds = np.where(det[:, 4].ravel() >= score_thresh)[0]
            det = det[inds]
            num_gt = gt_boxes.shape[0]
            num_det = det.shape[0]
            if num_det == 0:
                count_gt += num_gt
                continue
            ious = np.zeros((num_gt, num_det), dtype=np.float32)
            for i in range(num_gt):
                for j in range(num_det):
                    ious[i, j] = _compute_iou(gt_boxes[i], det[j, :4])
            tfmat = (ious >= iou_thresh)
            # for each det, keep only the largest iou of all the gt
            for j in range(num_det):
                largest_ind = np.argmax(ious[:, j])
                for i in range(num_gt):
                    if i != largest_ind:
                        tfmat[i, j] = False
            # for each gt, keep only the largest iou of all the det
            for i in range(num_gt):
                largest_ind = np.argmax(ious[i, :])
                for j in range(num_det):
                    if j != largest_ind:
                        tfmat[i, j] = False
            for j in range(num_det):
                y_score.append(det[j, -1])
                if tfmat[:, j].any():
                    y_true.append(True)
                else:
                    y_true.append(False)
            count_tp += tfmat.sum()
            count_gt += num_gt

        det_rate = count_tp * 1.0 / count_gt
        ap = average_precision_score(y_true, y_score) * det_rate

        precision, recall, __ = precision_recall_curve(y_true, y_score)
        recall *= det_rate
        return ap, det_rate

    def evaluate_search(self, gallery_det, gallery_feat, probe_feat,
                        score_thresh=0.5, base_iou_thresh=0.5,
                        gallery_size=100, dump_json=None):
        """
        gallery_det (list of ndarray): n_det x [x1, x2, y1, y2, score] per image
        gallery_feat (list of ndarray): n_det x D features per image
        probe_feat (list of ndarray): D dimensional features per probe image

        score_thresh (float): filter out gallery detections whose scores below this
        gallery_size (int): gallery size [-1, 50, 100, 500, 1000, 2000, 4000]
                            -1 for using full set
        dump_json (str): Path to save the results as a JSON file or None
        """
        assert self.num_images == len(gallery_det)
        assert self.num_images == len(gallery_feat)
        assert len(self.probes) == len(probe_feat)

        # TODO: support evaluation on training split
        use_full_set = gallery_size == -1
        fname = 'TestG{}'.format(gallery_size if not use_full_set else 50)
        protoc = loadmat(osp.join(self._root_dir, 'annotation/test/train_test',
                                  fname + '.mat'))[fname].squeeze()

        # mapping from gallery image to (det, feat)
        name_to_det_feat = {}
        for name, det, feat in zip(self._image_names, gallery_det, gallery_feat):
            scores = det[:, 4].ravel()
            inds = np.where(scores >= score_thresh)[0]
            if len(inds) > 0:
                name_to_det_feat[name] = (det[inds], feat[inds])

        aps = []
        accs = []
        topk = [1, 5, 10]
        recalls = []
        ret = {'image_root': self._data_path, 'results': []}
        for i in range(len(self.probes)):
            # y_true: {0, 1} labels of all detected gallery boxes, 1 means IoU(det_box, gt_probe_box) >=iou_thresh,
            # where gt_probe_box is the box with probe_id in the gallery image.
            # y_score: similarities between probe boxes and all detected gallery boxes
            y_true, y_score = [], []
            imgs, rois = [], []
            count_gt, count_tp = 0, 0
            # Get L2-normalized feature vector
            feat_p = probe_feat[i].ravel()
            # Ignore the probe image
            probe_imname = str(protoc['Query'][i]['imname'][0, 0][0])
            probe_roi = protoc['Query'][i]['idlocate'][0, 0][0].astype(np.int32)
            probe_roi[2:] += probe_roi[:2]
            probe_gt = []  # for vis
            tested = set([probe_imname])
            # 1. Go through the gallery samples defined by the protocol, which does not contain the probe image I guess.
            for item in protoc['Gallery'][i].squeeze():
                gallery_imname = str(item[0][0])
                # some contain the probe (gt not empty), some not
                gt = item[1][0].astype(np.int32)  # roi if this gallery image contain box of probe_id
                count_gt += (gt.size > 0)
                # compute distance between probe and gallery dets
                if gallery_imname not in name_to_det_feat: continue
                det, feat_g = name_to_det_feat[gallery_imname]
                # get L2-normalized feature matrix NxD
                assert feat_g.size == np.prod(feat_g.shape[:2])
                feat_g = feat_g.reshape(feat_g.shape[:2])
                # compute cosine similarities
                sim = feat_g.dot(feat_p).ravel()
                # assign label for each det
                label = np.zeros(len(sim), dtype=np.int32)
                if gt.size > 0:
                    w, h = gt[2], gt[3]
                    gt[2:] += gt[:2]
                    probe_gt.append({'img': str(gallery_imname),  # for vis
                                     'roi': list(map(float, list(gt)))})
                    iou_thresh = min(base_iou_thresh, (w * h * 1.0) / ((w + 10) * (h + 10)))
                    inds = np.argsort(sim)[::-1]
                    sim = sim[inds]
                    det = det[inds]
                    # only set the first matched det as true positive
                    for j, roi in enumerate(det[:, :4]):
                        if _compute_iou(roi, gt) >= iou_thresh:
                            label[j] = 1
                            count_tp += 1
                            break
                y_true.extend(list(label))
                y_score.extend(list(sim))
                imgs.extend([gallery_imname] * len(sim))
                rois.extend(list(det))
                tested.add(gallery_imname)
            # 2. Go through the remaining gallery images if using full set
            if use_full_set:
                for gallery_imname in self._image_names:
                    if gallery_imname in tested: continue
                    if gallery_imname not in name_to_det_feat: continue
                    det, feat_g = name_to_det_feat[gallery_imname]
                    # get L2-normalized feature matrix NxD
                    assert feat_g.size == np.prod(feat_g.shape[:2])
                    feat_g = feat_g.reshape(feat_g.shape[:2])
                    # compute cosine similarities
                    sim = feat_g.dot(feat_p).ravel()
                    # guaranteed no target probe in these gallery images
                    label = np.zeros(len(sim), dtype=np.int32)
                    y_true.extend(list(label))
                    y_score.extend(list(sim))
                    imgs.extend([gallery_imname] * len(sim))
                    rois.extend(list(det))
            # 3. Compute AP for this probe (need to scale by recall rate)
            y_score = np.asarray(y_score)
            y_true = np.asarray(y_true)
            assert count_tp <= count_gt
            recall_rate = count_tp * 1.0 / count_gt
            recalls.append(recall_rate)

            ap = 0 if count_tp == 0 else \
                average_precision_score(y_true, y_score) * recall_rate
            aps.append(ap)
            inds = np.argsort(y_score)[::-1]
            y_score = y_score[inds]
            y_true = y_true[inds]
            accs.append([min(1, sum(y_true[:k])) for k in topk])

            # 4. Save result for JSON dump
            new_entry = {'probe_img': str(probe_imname),
                         'probe_roi': list(map(float, probe_roi)),  # TODO(NOTE): list map
                         'probe_gt': probe_gt,
                         'gallery': []}
            # only save top-10 predictions
            for k in range(10):
                new_entry['gallery'].append({
                    'img': str(imgs[inds[k]]),
                    'roi': list(map(float, rois[inds[k]])),  # TODO(NOTE): list map
                    'score': float(y_score[k]),
                    'correct': int(y_true[k]),
                })
            ret['results'].append(new_entry)

        accs = np.mean(accs, axis=0)
        recall = np.mean(np.array(recalls), axis=0)

        if dump_json is not None:
            if not osp.isdir(osp.dirname(dump_json)):
                os.makedirs(osp.dirname(dump_json))
            with open(dump_json, 'w') as f:
                json.dump(ret, f)

        return np.mean(aps), topk, accs, recall

    def evaluate_cls(self, detections, pid_ranks, pid_labels, det_thresh=0.5):
        """
        detections (list of ndarray): n_det x [x1, x2, y1, y2, score] per image
        pid_ranks (list of ndarray): n_det x top_k cls scores per image
        pid_labels (list of ndarray): n_det x 1 ground truth identities

        det_thresh (float): filter out gallery detections whose scores below this
        """
        assert len(detections) == len(pid_ranks)
        assert len(detections) == len(pid_labels)

        # Get the num of identities in the imdb
        gt_roidb = self.fetch_gt_roidb()
        max_pid = 0
        for item in gt_roidb:
            max_pid = max(max_pid, max(item['gt_pids']))

        # In the extracted pid_labels:
        #   -1 for is_person_without_pid,
        #   {0, 1, ..., max_pid-1} for labeled person
        #   max_pid for background clutter
        count_ul, count_lb, count_bg = 0, 0, 0
        y_pred, y_true = [], []
        for dets, ranks, labels in zip(detections, pid_ranks, pid_labels):
            assert len(dets) == len(ranks)
            assert len(dets) == len(labels)
            for det, rank, label in zip(dets, ranks, labels):
                if det[-1] < det_thresh: continue
                label = int(round(label))
                if label == -1:
                    count_ul += 1
                    continue
                elif label == max_pid:
                    count_bg += 1
                    continue
                else:
                    count_lb += 1
                    y_pred.append(rank)
                    y_true.append(label)

        # some statistics
        print('classifiction:')
        print('  number of background clutter =', count_bg)
        print('  number of unlabeled =', count_ul)
        print('  number of labeled =', count_lb)

        # top-k classification accuracies
        correct = np.asarray(y_pred) == np.asarray(y_true)[:, np.newaxis]
        for top_k in [1, 5, 10]:
            acc = correct[:, :top_k].sum(axis=1).mean()
            print('  top-{} accuracy = {:.2%}'.format(top_k, acc))
