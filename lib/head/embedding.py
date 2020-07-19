import torch
from torch import nn
import torch.nn.functional as F


#################
### Detection ###
#################
class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models

    Arguments:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()
        self.representation_size = representation_size
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class EmbDet(nn.Module):
    def __init__(self, in_ch, out_ch, resolutions=[14, 7]):
        super(EmbDet, self).__init__()
        self.representation_size = out_ch

        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=2, padding=0),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True))
        self.resolutions = [resolutions[0] // 2, (resolutions[1] + 0 - 1) // 2 + 1]
        self.fc2 = nn.Linear(out_ch * self.resolutions[0] * self.resolutions[1], out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc2(x))
        return x


class CovEmbDet(nn.Module):
    """
    Takes RoI feature maps, and yields feature vectors without normalization.
    """

    def __init__(self, cov5, freeze_bn):
        super(CovEmbDet, self).__init__()
        self.representation_size = 2048
        self.freeze_bn = freeze_bn
        self.model = cov5
        self.pooling = nn.AdaptiveAvgPool2d(1)

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        if self.freeze_bn:
            self.model.apply(set_bn_fix)  # comment it for standard baseline

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        if mode:
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            if self.freeze_bn:
                self.model.apply(set_bn_eval)  # comment it for standard baseline
        return self

    def forward(self, x):
        x = self.model(x)
        box_feat = self.pooling(x).view(x.size(0), -1)
        return box_feat


#############
### re-ID ###
#############

class NormEmbReID(nn.Module):
    """
    Takes RoI feature maps, and yields feature vectors without normalization.
    """

    def __init__(self, top_model, freeze_bn, dim_emb=0, has_bn=True):
        super(NormEmbReID, self).__init__()
        self.dim_emb = dim_emb
        self.has_bn = has_bn
        self.out_channels = 2048
        self.freeze_bn = freeze_bn
        self.model = top_model
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.rescaler = nn.BatchNorm2d(1, affine=True)

        if dim_emb > 0:
            self.reduce_feat_dim = nn.Conv2d(self.out_channels, dim_emb, kernel_size=1)
            self.out_channels = dim_emb
        if has_bn:
            self.bn = nn.BatchNorm2d(self.out_channels)

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        if self.freeze_bn:
            self.model.apply(set_bn_fix)  # comment it for standard baseline

    @property
    def rescaler_weight(self):
        return self.rescaler.weight.item()

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        if mode:
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            if self.freeze_bn:
                self.model.apply(set_bn_eval)  # comment it for standard baseline
        return self

    def forward(self, x):
        ret = dict()
        embeddings = self.model(x)  # nchw
        if self.dim_emb:
            embeddings = self.reduce_feat_dim(embeddings)
        ret.update({"last_layer_feat": embeddings})
        if self.has_bn and embeddings.size(0) > 1:
            embeddings = self.bn(embeddings)

        norms = embeddings.norm(2, 1, keepdim=True)  # n1hw
        embeddings = embeddings / norms.clamp(min=1e-12)
        norms = self.rescaler(norms)  # n1hw

        spatial_attention = torch.sigmoid(norms)
        embeddings = F.adaptive_avg_pool2d(
            embeddings * spatial_attention, 1).flatten(start_dim=1)  # size = (N, d)

        ret.update({"feature": embeddings, "norms": norms})

        return ret


class EmbReID(nn.Module):
    """
    Takes RoI feature maps, and yields feature vectors without normalization.
    """

    def __init__(self, top_model, freeze_bn=False, dim_emb=-1, has_bn=True):
        super(EmbReID, self).__init__()
        self.dim_emb = dim_emb
        self.has_bn = has_bn
        self.out_channels = 2048
        self.freeze_bn = freeze_bn
        self.model = top_model
        self.model.add_module('pooling', nn.AdaptiveMaxPool2d(1))

        if dim_emb > 0:
            self.reduce_feat_dim = nn.Linear(self.out_channels, dim_emb, bias=False)
            self.out_channels = dim_emb
        if has_bn:
            self.bn = nn.BatchNorm1d(self.out_channels)

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        if self.freeze_bn:
            self.model.apply(set_bn_fix)

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        if mode:
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            if self.freeze_bn:
                self.model.apply(set_bn_eval)  # comment it for standard baseline
        return self

    def inference(self, x):
        ret = dict()
        box_feat = self.model(x).view(x.size(0), -1)
        if self.dim_emb > 0:
            box_feat = self.reduce_feat_dim(box_feat)
        if self.has_bn:
            box_feat = self.bn(box_feat)
        ret.update({"feature": box_feat})
        return ret

    def forward(self, x):
        if not self.training:
            return self.inference(x)
        ret = dict()
        box_feat = self.model(x).view(x.size(0), -1)
        if self.dim_emb > 0:
            box_feat = self.reduce_feat_dim(box_feat)
        ret.update({"last_layer_feat": box_feat})
        if self.has_bn and box_feat.size(0) > 1:
            box_feat = self.bn(box_feat)
        ret.update({"feature": box_feat})
        return ret


class MlpEmbReID(nn.Module):
    """
    Takes RoI feature maps, and yields feature vectors without normalization.
    """

    def __init__(self, in_ch, dim_emb=-1, has_bn=True):
        super(MlpEmbReID, self).__init__()
        self.dim_emb = dim_emb
        self.has_bn = has_bn
        self.out_channels = dim_emb if dim_emb > 0 else in_ch
        self.representation_size = self.out_channels

        if dim_emb > 0:
            self.reduce_feat_dim = nn.Linear(in_ch, dim_emb, bias=False)
        if has_bn:
            self.bn = nn.BatchNorm1d(self.out_channels)
            nn.init.normal_(self.bn.weight, std=0.01)
            nn.init.constant_(self.bn.bias, 0)

    def inference(self, box_feat):
        if self.dim_emb > 0:
            box_feat = self.reduce_feat_dim(box_feat)
        if self.has_bn:
            box_feat = self.bn(box_feat)
        return box_feat

    def forward(self, box_feat):
        if box_feat.dim() == 4:
            box_feat = box_feat.mean(-1).mean(-1)

        if not self.training:
            return self.inference(box_feat)
        if self.dim_emb > 0:
            box_feat = self.reduce_feat_dim(box_feat)
        if self.has_bn and box_feat.size(0) > 1:
            box_feat = self.bn(box_feat)
        return box_feat
