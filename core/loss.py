# -*- coding: utf-8 -*-

import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        pass


class WeightedSmoothL1Loss(nn.modules.loss.SmoothL1Loss):
    def __init__(self, reduction='elementwise_mean'):
        super().__init__(reduction='none')
        self.reduction = reduction

    def forward(self, input, target, weight):
        loss = super().forward(input, target)
        # dont need backward for weights,
        # it cames from input
        loss *= weight.detach()
        if self.reduction == 'elementwise_mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
