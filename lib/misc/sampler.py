import copy
import random
import numpy as np
from collections import defaultdict

import torch
from torch.utils.data.sampler import Sampler


def collate_fn():
    def collate(batch):
        ''' Alternative of default_collate function in torch.utils.data.dataloader.
            Assigning key for each item of the default_collate output

        Args:
            batch(list): contains the output of __getitem__ (a dict), termed args_dict. len(batch)=batch_size.

        Returns:

        '''

        assert isinstance(batch, list) and len(batch) > 0 and isinstance(batch[0], dict)
        img_list = []
        tgt_list = []
        for item in batch:
            img_list.append(item['image'])
            tgt_list.append(item['target'])

        ret = dict(images=img_list, targets=tgt_list)

        return ret

    return collate








