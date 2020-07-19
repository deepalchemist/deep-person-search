import math
import torch
from torch import nn
import torch.nn.functional as F


class OmniSoftMax(nn.Module):
    def __init__(self, num_features, num_classes, cls_type,
                 with_queue=False, l2_norm=False, scalar=1.0, momentum=0.5):
        super(OmniSoftMax, self).__init__()
        self.identify_mode = cls_type  # (sfm, oim)
        assert cls_type in ("sfm", "oim")
        self.num_features = num_features
        self.num_classes = num_classes
        self.momentum = momentum
        self.with_queue = with_queue
        self.l2_norm = l2_norm
        if "requires_grad" == scalar:  # requires_grad
            self.scalar = nn.Parameter(torch.tensor(1, dtype=torch.float).fill_(10.0), requires_grad=True)
        else:
            self.scalar = scalar

        if "sfm" == self.identify_mode:
            self.weight = nn.Parameter(torch.Tensor(num_classes, num_features), requires_grad=True)
            nn.init.normal_(self.weight, std=0.001)
        elif "oim" == self.identify_mode:
            assert self.l2_norm
            self.register_buffer('weight', torch.zeros(num_classes, num_features, requires_grad=False))
        else:
            raise NotImplementedError

        self.register_buffer('zero_loss', torch.zeros(1, requires_grad=False))
        if with_queue:
            self.register_buffer('unlabeled_input_count', torch.zeros(1, dtype=torch.int32, requires_grad=False))
            self.num_unlabeled = 5000 if num_classes > 1000 else 500
            self.register_buffer('unlabeled_queue',
                                 torch.zeros(self.num_unlabeled, num_features, requires_grad=False))

    def forward(self, inputs, targets):
        weight = self.weight
        if self.l2_norm:
            inputs = F.normalize(inputs, dim=1)  # TODO(bug) detach denominator causes NaN
            weight = F.normalize(weight, dim=1)

        if not self.with_queue:
            predicts = inputs.mm(weight.clone().t()) if "oim" == self.identify_mode else inputs.mm(weight.t())
            if "oim" == self.identify_mode:
                # Update
                for x, y in zip(inputs, targets):
                    self.weight[y] = self.momentum * self.weight[y] + (1. - self.momentum) * x.detach().clone()
                    self.weight[y] = F.normalize(self.weight[y], dim=0)
            output_targets = targets
        else:
            labeled_inputs = inputs[targets > -1]
            labeled_targets = targets[targets > -1]
            unlabeled_inputs = inputs[targets == -1]

            # Counts valid unlabeled input
            self.unlabeled_input_count += unlabeled_inputs.size(0)
            self.unlabeled_input_count.fill_(min(self.num_unlabeled, self.unlabeled_input_count.item()))
            # Update the unlabeled queue before calculating loss, so that bbox features inside the same
            # image can compete with each other. These features are already l2-normalized
            self.unlabeled_queue = torch.cat([self.unlabeled_queue, unlabeled_inputs.detach().clone()])[
                                   -self.num_unlabeled:]
            if (targets > -1).sum().item() == 0:
                return None, None
            valid_unlabeled_queue = self.unlabeled_queue[-self.unlabeled_input_count.item():] \
                if self.unlabeled_input_count > 0 else self.unlabeled_queue[0:0]

            predicts = labeled_inputs.mm(torch.cat([weight.clone(), valid_unlabeled_queue]).t()) \
                if "oim" == self.identify_mode else \
                labeled_inputs.mm(torch.cat([weight, valid_unlabeled_queue]).t())
            # Update
            if "oim" == self.identify_mode:
                self.weight[labeled_targets] = self.momentum * self.weight[labeled_targets] + \
                                               (1. - self.momentum) * labeled_inputs.detach().clone()
                self.weight[labeled_targets] = F.normalize(self.weight[labeled_targets], p=2, dim=1)
            output_targets = labeled_targets

        predicts = predicts * self.scalar if self.l2_norm else predicts
        return predicts, output_targets


if __name__ == '__main__':
    def test(**kwargs):
        cls = OmniSoftMax(**kwargs)
        input = torch.rand([5, 100], requires_grad=True)
        target = torch.tensor([0, 1, 2, 3, -1], dtype=torch.long)
        crit = torch.nn.CrossEntropyLoss()

        output, target = cls(input, target)
        loss = crit(output, target)
        print(loss)
        loss.backward()


    kwargs = dict(identify_mode='sfm', num_features=100, num_classes=10,
                  scalar=1.0, momentum=0.5, with_queue=True, l2_norm=False)
    test(**kwargs)
