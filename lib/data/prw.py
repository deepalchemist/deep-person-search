import os
import sys
import cv2
import json
from PIL import Image
import os.path as osp
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import average_precision_score, precision_recall_curve

from tqdm import tqdm
from dps.data.imdb import imdb


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    union_area = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter_area
    return inter_area * 1.0 / union_area


def get_pos_detection(det, gt_boxes, iou_thresh=0.5):
    y_true = []

    num_gt = gt_boxes.shape[0]
    num_det = det.shape[0]
    ious = np.zeros((num_gt, num_det), dtype=np.float32)  # 2D array, (num_gt, num_det)
    for i in range(num_gt):
        for j in range(num_det):
            ious[i, j] = compute_iou(gt_boxes[i], det[j, :4])

    # tfmat(2D array): (num_gt, num_det)
    tfmat = (ious >= iou_thresh)
    # for each det, keep only the largest iou of all the gt
    for j in range(num_det):
        largest_ind = np.argmax(ious[:, j])  # ranking by iou
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
        if tfmat[:, j].any():
            y_true.append(True)
        else:
            y_true.append(False)
    return np.array(y_true)


def write_detected_gt_boxes(img_path, det, gt_boxes, gt_pid, out_dir='./cache/detections/prw/'):
    assert gt_boxes.shape[0] == gt_pid.shape[0], print(gt_boxes.shape, gt_pid.shape)
    det = np.round(det).astype(np.int)
    gt_pid = gt_pid.astype(np.int)

    cv2im = cv2.imread(img_path)  # (height, width, nchannel), 'BGR'
    file_name = os.path.basename(img_path)
    for (x1, y1, x2, y2), pid in zip(gt_boxes, gt_pid):
        # import pdb;pdb.set_trace()
        ec = (0, 255, 0)  # (green)
        cv2im = cv2.rectangle(cv2im, (x1, y1 - 16), (x2, y1), ec,
                              thickness=-1)  # thickness=negative means filled rectangle

        cv2im = cv2.putText(cv2im, "{:d}".format(pid), org=(x1 + 4, y1 - 2),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 255, 255),
                            thickness=1, lineType=cv2.LINE_AA)
        cv2im = cv2.rectangle(cv2im, (x1, y1), (x2, y2), ec, thickness=1)

    for x1, y1, x2, y2 in det:
        ec = (255, 0, 0)  # (blue)
        cv2im = cv2.rectangle(cv2im, (x1, y1), (x2, y2), ec, thickness=1)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Make dir: {}".format(out_dir))
    cv2.imwrite(out_dir + file_name, cv2im)
    return cv2im


class prw(imdb):
    '''
    labels its ids from 1 to 933, -2 means unsure, 11816 images in total
    num_train_id = 483 (not include -2), num_test_id = 450 (not include -2)
    len(frame_train)=5704, where 40 images contains only -2 boxes and 5664 images contain at lease an id-labeled box
    len(frame_test)=6112, where 46 images contains only -2 boxes and 6066 images contain at lease an id-labeled box
    number ids in frame_train: 484 (include -2), number ids in frame_test: 545 (include -2 and some training ids)
    #query=2057, note that query images are included in frame_test
    #train_boxes=14907(not include 3141 unlabeled), #query_boxes=2057,
    #gallery_boxes=23005(23002 remained due to error id label)
    (c1, c4), (c2, c3, c5)
    '''

    def __init__(self, image_set, data_type='origin', probe_type='origin', use_mask=False, root_dir=None):
        super(prw, self).__init__('prw_' + image_set)
        self._image_set = image_set
        self._root_dir = os.path.join(root_dir, 'prw')
        self.data_path = data_type
        assert osp.isdir(self._root_dir), "Path does not exist: {}".format(self._root_dir)
        assert osp.isdir(self._data_path), "Path does not exist: {}".format(self._data_path)

        self._id2idx = self._train_id_to_index()
        self._num_train_ids = len(self.id2idx)
        self._image_names = self._load_split_img_name()
        self.probes = probe_type
        self._roidb = self.fetch_gt_roidb()

        use_mask = use_mask
        if use_mask:
            # Binary masks of pedestrians, in (h, w), {0, 255}
            mask_dir = osp.join(self._root_dir, 'mask')
            self.set_mask_dir(mask_dir)
            assert osp.isdir(self._mask_dir), "Path does not exist: {}".format(self._mask_dir)
            self.add_mask_to_roidb()

    @property
    def data_path(self):
        return self._data_path

    @data_path.setter
    def data_path(self, type='origin'):
        data_type = {
            "origin": "frames",
        }
        self._data_path = osp.join(self._root_dir, data_type[type])

    @property
    def probes(self):
        return self._probes

    @probes.setter
    def probes(self, type='origin'):
        probe_type = {
            "origin": self._load_probes,
            "mini": self._load_mini_probes
        }
        assert type in probe_type
        self._probes = probe_type[type]()

    def _train_id_to_index(self):
        # return dict has int keys and int values
        train_ids = loadmat(osp.join(self._root_dir, 'ID_train.mat'))
        train_ids = train_ids['ID_train'][0]  # already sorted, (1 483) to (483)
        id2idx = dict()
        for ii, pid in enumerate(train_ids):
            id2idx[pid] = ii
        return id2idx

    def _load_split_img_name(self):
        """
        Load the file names for the specific subset (train / test).
        Note that the file name has extension (e.g., .jpg).
        """
        # test pool
        test = loadmat(osp.join(self._root_dir, 'frame_test.mat'))
        test = test['img_index_test'].squeeze()
        test = [str(test_image[0]) + '.jpg' for test_image in test]
        if self._image_set == 'test': return test

        # train pool
        train = loadmat(osp.join(self._root_dir, 'frame_train.mat'))
        train = train['img_index_train'].squeeze()
        train = [str(train_image[0]) + '.jpg' for train_image in train]
        # training
        return train

    def _load_probes(self):
        # line 563: 605 -1.7030 295.5756 74.6436 338.2289 c3s2_044587
        # line 1436: 791 -1.7030 423.8693 247.2570 646.1339 c1s4_048156
        probes = []
        file = open(osp.join(self._root_dir, 'query_info.txt'), "r")
        for line_no, line in enumerate(file):
            data = line.split()
            ID = int(data[0])
            probe_roi = np.array([float(data[1]), float(data[2]), float(data[3]), float(data[4])])
            probe_im_name = data[5]
            probe_im_name = osp.join(self._data_path, probe_im_name + ".jpg")
            probe_roi = probe_roi.astype(np.int32)
            probe_roi[2:] += probe_roi[:2]
            probes.append((probe_im_name, probe_roi, ID))  # set fixed query box

        return probes

    def _load_mini_probes(self):
        print("* Evaluating with mini_prw_probe.")
        probes = []
        file = open(osp.join(self._root_dir, 'prwmini_query_info.txt'), "r")
        for line_no, line in enumerate(file):
            data = line.split()
            ID = int(data[0])
            probe_roi = np.array([float(data[1]), float(data[2]), float(data[3]), float(data[4])])
            probe_im_name = data[5]
            probe_im_name = osp.join(self._data_path, probe_im_name + ".jpg")
            probe_roi = probe_roi.astype(np.int32)
            probe_roi[2:] += probe_roi[:2]
            probes.append((probe_im_name, probe_roi, ID))  # set fixed query box

        return probes

    def fetch_gt_roidb(self):
        gt_roidb = []
        for index, im_name in enumerate(self._image_names):
            boxes_pid = loadmat(osp.join(self._root_dir, 'annotations',
                                         im_name + '.mat'))
            if 'box_new' in boxes_pid:
                boxes_pid = boxes_pid['box_new']
            elif 'anno_file' in boxes_pid:
                boxes_pid = boxes_pid['anno_file']
            else:
                boxes_pid = boxes_pid['anno_previous']

            boxes = boxes_pid[:, 1:5]
            boxes = boxes.astype(np.int32)

            boxes[:, 2] += boxes[:, 0]  # x axis, W
            boxes[:, 3] += boxes[:, 1]  # y axis, H

            pids = boxes_pid[:, 0]
            # re-labeling
            if self._image_set == 'train':
                new_pids = []
                for pid in pids:
                    if pid == -2:  # is_person_unsure_pid
                        new_pids.append(-1)  # Note: assign is_person_unsure_pid as -1
                    else:
                        new_pids.append(self.id2idx[pid])  # Note: label person from 0 to N-1
                pids = np.array(new_pids, dtype=np.long)

            num_objs = len(boxes)
            gt_classes = np.ones((num_objs), dtype=np.int32)

            img_size = Image.open(self.image_path_at(index)).size

            gt_roidb.append({
                # scene image
                'img_name': im_name,
                'img_path': self.image_path_at(index),
                'width': img_size[0],
                'height': img_size[1],
                'flipped': False,
                # boxes
                'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_pids': pids,
            })

        return gt_roidb

    def evaluate_detections(self, gallery_det, score_thresh=0.5, iou_thresh=0.5, labeled_only=False):
        """
        gallery_det (list of ndarray): n_det x [x1, x2, y1, y2, score] per image
        score_thresh (float): filter out gallery detections whose scores below this
        iou_thresh (float): treat as true positive if IoU is above this threshold
        labeled_only (bool): filter out unlabeled background people
        """
        assert self.num_images == len(gallery_det)

        gt_roidb = self._roidb
        y_true, y_score = [], []
        count_gt, count_tp = 0, 0
        # go through gallery images
        n_iter = 0
        for gt, det in zip(gt_roidb, gallery_det):

            ### single image records ###
            gt_pids = gt['gt_pids']
            ############################

            gt_boxes = gt['boxes']
            if labeled_only:  # only use id-labeled bounding box for test
                inds = np.where(gt['gt_pids'].ravel() > 0)[0]
                if len(inds) == 0: continue
                gt_boxes = gt_boxes[inds]
                gt_pids = gt_pids[inds]

            det = np.asarray(det)
            # 1. keep detections whose cls_score >= score_thresh
            inds = np.where(det[:, 4].ravel() >= score_thresh)[0]
            det = det[inds]

            num_gt = gt_boxes.shape[0]
            num_det = det.shape[0]
            if num_det == 0:
                count_gt += num_gt
                continue
            ious = np.zeros((num_gt, num_det), dtype=np.float32)  # 2D array, (num_gt, num_det)
            for i in range(num_gt):
                for j in range(num_det):
                    ious[i, j] = compute_iou(gt_boxes[i], det[j, :4])

            # 2. iou_thresh, tfmat(2D array): (num_gt, num_det)
            tfmat = (ious >= iou_thresh)
            # for each det, keep only the largest iou of all the gt
            for j in range(num_det):
                largest_ind = np.argmax(ious[:, j])  # ranking by iou
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

            # y_true(list): len equals num_det, it indicates positive or negative detections
            count_tp += tfmat.sum()
            count_gt += num_gt
            n_iter += 1

        det_rate = count_tp * 1.0 / count_gt  # recall

        # todo(note): the same recall but lower ap indicates bbox has high iou but low class score.
        ap = average_precision_score(y_true, y_score) * det_rate

        precision, recall, __ = precision_recall_curve(y_true, y_score)  # TODO: use for what
        recall *= det_rate

        return ap, det_rate

    def evaluate_search(self, gallery_det, gallery_feat, probe_feat,
                        score_thresh=0.5, base_iou_thresh=0.5, dump_json=None):

        gallery_det = list(map(lambda x: np.array(x), gallery_det))
        gallery_feat = list(map(lambda x: np.array(x), gallery_feat))
        # gallery image name to its detected bboxes and bbox features
        name_to_box_and_feat = {}
        for entry, det, feat in tqdm(zip(self.roidb, gallery_det, gallery_feat),
                                     desc="filtering positive detections"):
            name = entry['img_name']
            pids = entry['gt_pids']
            # 1. filter detections with classification score
            scores = det[:, 4].ravel()
            inds = np.where(scores >= score_thresh)[0]
            # if len(inds) > 0:
            #     name_to_box_and_feat[name] = (det[inds], feat[inds], pids)
            if not len(inds) > 0:
                continue
            det, feat = det[inds], feat[inds]

            ## 2. get positive detections todo(note): only used for analysis
            # keeps = get_pos_detection(det, entry['boxes'], iou_thresh=base_iou_thresh)
            # if keeps.sum() == 0:
            #     continue
            # det, feat = det[keeps], feat[keeps]

            name_to_box_and_feat[name] = (det, feat, pids)

        aps = []
        accs = []
        topk = [1, 5, 10]
        recalls = []
        ret = {'image_root': self._data_path, 'results': []}
        # go through probe bounding boxes
        # _t = {'single_gallery': util.Timer(), 'iou': util.Timer()}
        for i in tqdm(range(len(self.probes))):
            y_true, y_score = [], []
            imgs, rois = [], []
            count_gt, count_tp = 0, 0

            feat_p = probe_feat[i].ravel()[:, np.newaxis]
            # feat_p = normalize(feat_p, axis=0)  # (256,1)

            probe_imname = self.probes[i][0].split('/')[-1]
            probe_roi = self.probes[i][1]
            probe_pid = self.probes[i][2]

            # gather image that contains probe_pid but exclude the probe image
            gallery_imgs = filter(lambda x: probe_pid in x['gt_pids'] and x['img_name'] != probe_imname, self.roidb)
            probe_gts = {}  # {img_name of a gallery img that contains probe_pid: true_positive box coordinate}
            for item in gallery_imgs:
                probe_gts[item['img_name']] = item['boxes'][
                    item['gt_pids'] == probe_pid]

            tested = set([probe_imname])

            # Select the gallery set, which only exclude the probe image
            gallery_imgs = filter(lambda x: x['img_name'] != probe_imname, self.roidb)
            # Go through the selected gallery
            for item in gallery_imgs:
                gallery_imname = item['img_name']
                # some contain the probe (gt not empty), some not
                count_gt += (gallery_imname in probe_gts)
                # compute distance between probe and gallery dets
                if gallery_imname not in name_to_box_and_feat:
                    continue
                det, feat_g, pids_g = name_to_box_and_feat[gallery_imname]

                # get L2-normalized feature matrix NxD
                assert feat_g.size == np.prod(feat_g.shape[:2])

                feat_g = feat_g.reshape(feat_g.shape[:2])
                # feat_g = normalize(feat_g, axis=1)

                # compute cosine similarities
                sim = feat_g.dot(feat_p).ravel()
                # assign label for each det
                label = np.zeros(len(sim), dtype=np.int32)

                if gallery_imname in probe_gts:
                    gt = probe_gts[gallery_imname].ravel()
                    w, h = gt[2] - gt[0], gt[3] - gt[1]
                    iou_thresh = min(base_iou_thresh, (w * h * 1.0) /
                                     ((w + 10) * (h + 10))
                                     )
                    inds = np.argsort(sim)[::-1]  # todo-note: sorted via similarity
                    sim = sim[inds]
                    det = det[inds]
                    # only set the first matched det as true positive
                    for j, roi in enumerate(det[:, :4]):
                        # in the tgt gallery img, only set one roi as true positive
                        # todo(note): low mAP indicates high similarity but iou less than iou_thresh? somewhat
                        if compute_iou(roi, gt) >= iou_thresh:
                            label[j] = 1
                            count_tp += 1
                            break

                y_true.extend(list(label))
                y_score.extend(list(sim))
                imgs.extend([gallery_imname] * len(sim))
                rois.extend(list(det))
                tested.add(gallery_imname)

            # sys.stdout.write('go through probes: {:d}/{:d} single gallery:{:.3f}s iou:{:.3f}s   \r'.format(
            #     i + 1, len(self.probes), _t['single_gallery'].average_time, _t['iou'].average_time))
            # sys.stdout.flush()

            # 3. Compute AP for this probe (need to scale by recall rate)
            y_score = np.asarray(y_score)
            y_true = np.asarray(y_true)
            assert count_tp <= count_gt
            recall_rate = count_tp * 1.0 / count_gt
            recalls.append(recall_rate)
            # print("* >>>>>>>>>>> recall: {:.2f}".format(recall_rate))

            ap = 0 if count_tp == 0 else \
                average_precision_score(y_true, y_score) * recall_rate
            aps.append(ap)
            inds = np.argsort(y_score)[::-1]
            y_true = y_true[inds]
            accs.append([min(1, sum(y_true[:k])) for k in topk])

            # 4. Save result for JSON dump
            probe_gt = [{'img': str(k), 'roi': list(map(float, list(v.flat)))} for k, v in probe_gts.items()]
            new_entry = {'probe_img': str(probe_imname),
                         'probe_roi': list(map(float, probe_roi)),  # todo(note): list map
                         'probe_gt': probe_gt,
                         'gallery': []}
            # only save top-10 predictions
            for k in range(10):
                new_entry['gallery'].append({
                    'img': str(imgs[inds[k]]),
                    'roi': list(map(float, rois[inds[k]])),  # todo(note): list map
                    'score': float(y_score[k]),
                    'correct': int(y_true[k]),
                })
            ret['results'].append(new_entry)

        mAP = np.mean(aps)
        accs = np.mean(accs, axis=0)
        recall = np.mean(np.array(recalls), axis=0)

        if dump_json is not None:
            if not osp.isdir(osp.dirname(dump_json)):
                os.makedirs(osp.dirname(dump_json))
            with open(dump_json, 'w') as f:
                json.dump(ret, f)

        return mAP, topk, accs, recall
