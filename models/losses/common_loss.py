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
        batch_size = weight.shape[0]

        # how to reshape them
        preds_shape = preds.shape
        target_shape = target.shape
        if len(preds_shape) == len(target_shape):
            # assume one2one match(reg loss)
            loss = forward(self, preds, target) * weight.unsqueeze(-1)
            loss = loss.sum(dim=-1).sum(dim=-1)

        elif len(preds_shape) == len(target_shape) + 1:
            # assume cls loss
            weight = weight.view(-1)
            target = target.view(-1)
            preds = preds.view(-1, preds_shape[-1])

            loss = forward(self, preds, target) * weight
            loss = loss.view(batch_size, -1).sum(dim=-1)
        else:
            raise ValueError('can not assume any possible loss type')

        return loss

    cls.forward = new_forward

    return cls


CrossEntropyLoss = WeightedLossWrapper(nn.CrossEntropyLoss)
SmoothL1Loss = WeightedLossWrapper(nn.SmoothL1Loss)
