import os
import PIL
import numpy as np


class imdb(object):
    """Image database."""
    classes = ('__background__', 'person')

    def __init__(self, name):
        self._name = name
        self._data_path = ''
        self._id2idx = {}
        self._num_train_ids = 0
        self._image_names = []
        self._probes = []
        self._roidb = []

    @property
    def name(self):
        return self._name

    @property
    def id2idx(self):
        return self._id2idx

    @property
    def num_train_ids(self):
        return self._num_train_ids

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def image_names(self):
        return self._image_names

    @property
    def num_images(self):
        return len(self.image_names)

    @property
    def roidb(self):
        return self._roidb

    @roidb.setter
    def roidb(self, _roidb):
        self._roidb = _roidb

    @property
    def cache_path(self):
        cache_path = os.path.abspath(os.path.join('./', 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    def image_path_at(self, i):
        """
        Return the absolute path of i-th image in the image sequence.
        """
        image_path = os.path.join(self._data_path, self._image_names[i])
        assert os.path.isfile(image_path), "Path does not exist: {}".format(image_path)
        return image_path

    def set_mask_dir(self, dir):
        self._mask_dir = dir

    def mask_path_at(self, i):
        """
        Return the absolute mask path of i-th image in the sequence.
        """
        mask_path = os.path.join(self._mask_dir, self._image_names[i])
        assert os.path.isfile(mask_path), "Mask does not exist: {}".format(mask_path)
        return mask_path

    def add_mask_to_roidb(self):
        for ii, item in enumerate(self._roidb):
            new = {'mask_path': self.mask_path_at(ii)}
            item.update(new)

    def image_id_at(self, i):
        raise NotImplementedError

    def _train_id_to_index(self):
        raise NotImplementedError

    def _load_split_img_name(self):
        raise NotImplementedError

    def fetch_gt_roidb(self):
        raise NotImplementedError

    def evaluate_detections(self,
                            all_boxes,
                            output_dir=None):
        """
        all_boxes is a list of length number-of-classes.
        Each list element is a list of length number-of-images.
        Each of those list elements is either an empty list []
        or a numpy array of detection.

        all_boxes[class][image] = [] or np.array of shape #dets x 5
        """
        raise NotImplementedError

    def evaluate_search(self,
                        gallery_det,
                        gallery_feat,
                        probe_feat):
        raise NotImplementedError

    def append_flipped_images(self):
        num_images = self.num_images
        for i in range(num_images):
            entry = {k: v for k, v in self.roidb[i].items()}  # copy
            width = entry['width']
            # flip boxes
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = width - oldx2 - 1
            boxes[:, 2] = width - oldx1 - 1
            assert (boxes[:, 2] >= boxes[:, 0]).all()
            # update
            entry['boxes'] = boxes
            entry['flipped'] = True
            # append flipped
            self.roidb.append(entry)
        self._image_names = self._image_names * 2

    @staticmethod
    def rank_roidb_ratio(roidb):
        # rank roidb based on the ratio between width and height.
        ratio_large = 2  # largest ratio to preserve.
        ratio_small = 0.5  # smallest ratio to preserve.

        ratio_list = []
        for i in range(len(roidb)):
            width = roidb[i]['width']
            height = roidb[i]['height']
            ratio = width / float(height)

            if ratio > ratio_large:
                roidb[i]['need_crop'] = 1
                ratio = ratio_large
            elif ratio < ratio_small:
                roidb[i]['need_crop'] = 1
                ratio = ratio_small
            else:
                roidb[i]['need_crop'] = 0

            ratio_list.append(ratio)

        ratio_list = np.array(ratio_list)
        ratio_index = np.argsort(ratio_list)
        return ratio_list[ratio_index], ratio_index

    @staticmethod
    def prepare_roidb(imdb):
        """Enrich the imdb's roidb by adding some derived quantities that
        are useful for training. This function precomputes the maximum
        overlap, taken over ground-truth boxes, between each ROI and
        each ground-truth box. The class with maximum overlap is also
        recorded.
        """
        if not (imdb.name.startswith('coco')):
            sizes = [PIL.Image.open(imdb.image_path_at(i)).size
                     for i in range(imdb.num_images)]

        for i in range(len(imdb.image_names)):
            imdb.roidb[i]['img_id'] = imdb.image_id_at(i)
            imdb.roidb[i]['img_path'] = imdb.image_path_at(i)
            if not (imdb.name.startswith('coco')):
                imdb.roidb[i]['width'] = sizes[i][0]
                imdb.roidb[i]['height'] = sizes[i][1]
            # need gt_overlaps as a dense array for argmax
            gt_overlaps = imdb.roidb[i]['gt_overlaps'].toarray()
            # max overlap with gt over classes (columns)
            max_overlaps = gt_overlaps.max(axis=1)
            # gt class that had the max overlap
            max_classes = gt_overlaps.argmax(axis=1)
            imdb.roidb[i]['max_classes'] = max_classes
            imdb.roidb[i]['max_overlaps'] = max_overlaps
            # sanity checks
            # max overlap of 0 => class should be zero (background)
            zero_inds = np.where(max_overlaps == 0)[0]
            assert all(max_classes[zero_inds] == 0)
            # max overlap > 0 => class should not be zero (must be a fg class)
            nonzero_inds = np.where(max_overlaps > 0)[0]
            assert all(max_classes[nonzero_inds] != 0)


if __name__ == '__main__':
    class inst(imdb):
        def __init__(self, name):
            super(inst, self).__init__(name)


    a = inst('prw')
    print(a.classes)
