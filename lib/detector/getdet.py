from collections import OrderedDict

import torch.nn as nn


class DetectorBackbone(nn.Module):
    def __init__(self, base_model, return_layers):
        super(DetectorBackbone, self).__init__()
        for n, k in base_model.named_children():
            self.add_module(n, k)
        self.return_layers = return_layers
        self.out_channels = 1024

        # Fix blocks
        for p in self.conv1.parameters(): p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.apply(set_bn_fix)

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        if mode:
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.apply(set_bn_eval)
        return self

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out
