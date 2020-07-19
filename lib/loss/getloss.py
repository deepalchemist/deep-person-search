import torch

from .softmax import OmniSoftMax

from lib.misc import util

_factory = {
    "sfm": OmniSoftMax,
    "oim": OmniSoftMax,
}


def init_classifier(cls_type, feat_dim, num_pid):
    kwargs = dict(num_features=feat_dim, num_classes=num_pid)
    classifier = _factory[cls_type]
    if cls_type == 'sfm' or cls_type == 'oim':
        kwargs.update(dict(
            cls_type=cls_type,
            with_queue=False if "sfm" == cls_type else True,
            l2_norm=False if "sfm" == cls_type else True,
            scalar=1. if "sfm" == cls_type else 30.,
        ))
    classifier = classifier(**kwargs)

    return classifier


class CriterionReID(torch.nn.Module):
    def __init__(self,
                 cls_type,
                 feat_dim,
                 num_train_pids,
                 ):
        super(CriterionReID, self).__init__()
        self.cls_type = cls_type
        self.feat_dim = feat_dim
        # Config classifier
        self.classifier = init_classifier(cls_type, feat_dim, num_train_pids)
        self.criterion_ide = torch.nn.CrossEntropyLoss()

    def forward(self, features, pids=None):
        """
        Arguments:
            features (2D Tensor):(* dim_feat), with pid >= 0, i.e. id-labeled person, (n c h w)
            pids (List[Tensor[N]]): not used in inference
        """
        if not self.training:
            return 0., 0.

        # cross-entropy loss
        pids = torch.cat(pids, dim=0)
        pred, cls_out_tgt = self.classifier(features, pids)
        loss_ide = 0.5 * self.criterion_ide(pred, cls_out_tgt)
        acc = util.classifier_accuracy(pred, cls_out_tgt)
        return loss_ide, acc
