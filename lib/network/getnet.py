import torch
from torch import nn
from torchvision.models import resnet
from torchvision.ops import misc as misc_nn_ops

from lib.network.ResNet import resnet50, resnet101

network_factory = {"resnet50": resnet50,
                   "resnet101": resnet101}
pretrained_weight = {"resnet50": './cache/pretrained_model/resnet50_caffe.pth',
                     "resnet101": './cache/pretrained_model/resnet101_caffe.pth'}


def _split_backbone(backbone_name, load_bgr=True, **kwargs):
    base_model = network_factory[backbone_name](**kwargs)
    if load_bgr:
        weight_path = pretrained_weight[backbone_name]
        print("* Backbone loading ImageNet pretrained weights from \n\t%s".expandtabs(4) % (weight_path))
        state_dict = torch.load(weight_path)
        base_model.load_state_dict({k: v for k, v in state_dict.items() if k in base_model.state_dict()})

    # separate from cov4_4
    depth_layer3 = len(base_model.layer3)
    layer3_bottom = nn.Sequential(*[base_model.layer3[i] for i in range(0, depth_layer3 // 2)])
    layer3_top = nn.Sequential(*[base_model.layer3[i] for i in range(depth_layer3 // 2, depth_layer3)])

    base = nn.Sequential()
    base.add_module('conv1', nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu, base_model.maxpool))
    base.add_module('conv2', base_model.layer1)
    base.add_module('conv3', base_model.layer2)
    base.add_module('conv4_3', layer3_bottom)

    top = nn.Sequential()
    top.add_module('conv4_4', layer3_top)
    top.add_module('conv5', base_model.layer4)
    return base, top


def split_backbone(backbone_name, load_bgr=True, freeze_top_bn=False):
    backbone = resnet.__dict__[backbone_name](
        pretrained=not load_bgr,
        norm_layer=misc_nn_ops.FrozenBatchNorm2d)

    if load_bgr:
        weight_path = pretrained_weight[backbone_name]
        print("* Backbone loading ImageNet pretrained weights from \n\t%s".expandtabs(4) % (weight_path))
        state_dict = torch.load(weight_path, map_location=torch.device("cpu"))
        backbone.load_state_dict({k: v for k, v in state_dict.items() if k in backbone.state_dict()})

    # freeze cov1 layers
    for name, parameter in backbone.named_parameters():
        if 'layer1' not in name and 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
            parameter.requires_grad_(False)

    # separate from cov4_4
    depth_layer3 = len(backbone.layer3)
    layer3_bottom = nn.Sequential(*[backbone.layer3[i] for i in range(0, depth_layer3 // 2)])
    layer3_top = nn.Sequential(*[backbone.layer3[i] for i in range(depth_layer3 // 2, depth_layer3)])

    base = nn.Sequential()
    base.add_module('conv1', nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool))
    base.add_module('conv2', backbone.layer1)
    base.add_module('conv3', backbone.layer2)
    base.add_module('conv4_3', layer3_bottom)
    base.out_channels = 1024

    top = nn.Sequential()
    top.add_module('conv4_4', layer3_top)
    top.add_module('conv5', backbone.layer4)
    top.add_module('pool6', nn.AdaptiveAvgPool2d(1))
    top.representation_size = 2048

    if not freeze_top_bn:
        top.apply(convert_fbn_to_bn)

    return base, top


def convert_fbn_to_bn(model):
    def copy_params(old_module, new_module):
        new_module.load_state_dict(
            {k: v for k, v in old_module.state_dict().items()
             if k in new_module.state_dict()}
        )
        return new_module

    for child_name, child in model.named_children():
        if isinstance(child, misc_nn_ops.FrozenBatchNorm2d):
            new_bn = nn.BatchNorm2d(child.weight.size(0))
            new_bn = copy_params(child, new_bn)
            setattr(model, child_name, new_bn)
        else:
            convert_fbn_to_bn(child)


if __name__ == '__main__':
    base, top = split_backbone('resnet50', load_bgr=False)
    print(base, top)
    # top.apply(convert_fbn_to_bn)
    # print(top)
