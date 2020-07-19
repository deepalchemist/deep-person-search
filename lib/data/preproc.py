import numpy as np
from dps.data.ssm import ssm
from dps.data.prw import prw

__all__ = ['get_imdb']

__sets = {'ssm': ssm,
          'prw': prw}


def get_imdb(name, **kwargs):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name](**kwargs)


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())


def filter_roidb(roidb):
    # filter the image without bounding box or id_annotated_box.
    num_img_before = len(roidb)
    i = 0
    while i < len(roidb):
        if len(roidb[i]['boxes']) == 0 or np.all(roidb[i]['gt_pids'] < 0):  # todo(note)
            del roidb[i]
            i -= 1
        i += 1
    print('* Before/After filtering, there are %d/%d images.' % (num_img_before, len(roidb)))
    return roidb


def combine_database():
    return


def create_data(imdb_names, data_root, no_flip, training=True, **kwargs):
    """roidb: region of interest database"""

    def get_single_imdb(imdb_name, **kwargs):
        imdb = get_imdb(imdb_name, **kwargs)
        print('Loaded dataset `{:s}` for {}ing'.format(imdb.name, kwargs['image_set']))
        if not no_flip:
            print('Appending horizontally-flipped training examples.')
            imdb.append_flipped_images()
        return imdb

    # args for initializing imbd
    kwargs.update(dict(
        image_set='train' if training else 'test',
        root_dir=data_root
    ))

    names = imdb_names.split('+')

    if not training:
        # Only support single dataset evaluation.
        # If takes multiple names, then uses the first one for test.
        return get_single_imdb(names[0], **kwargs)

    if len(names) > 1:
        raise NotImplementedError
    else:
        imdb = get_single_imdb(names[0], **kwargs)
        imdb.roidb = filter_roidb(imdb.roidb)
        return imdb


if __name__ == '__main__':
    imdb = create_data('prw', False, '/mnt/data2/caffe/person_search')
    print(len(imdb.roidb))
