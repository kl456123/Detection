# -*- coding: utf-8 -*-
"""
Wrap pytorch loss
"""
import torch.nn as nn


def WeightedLossWrapper(cls):

    forward = cls.forward

    def new_forward(self, targets):
        """
        Args:
            weight: shape(N)
            pred: shape(N, num_classes)
            target: shape(N)
        """
        weight = targets['weight']
        preds = targets['pred']
        target = targets['target']
        loss = forward(self, preds, target) * weight
        return loss

    cls.forward = new_forward

    return cls


CrossEntropyLoss = WeightedLossWrapper(nn.CrossEntropyLoss)
SmoothL1Loss = WeightedLossWrapper(nn.SmoothL1Loss)
