# -*- coding: utf-8 -*-
"""
Wrap pytorch loss
"""
import torch.nn as nn
import torch


# TODO (bugs of normalize=True)
# lr too small for sgd
def calc_loss(module, targets, normalize=True):
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
        loss = module(preds, target) * weight.unsqueeze(-1)
        loss = loss.sum(dim=-1)

    elif len(preds_shape) == len(target_shape) + 1:
        # assume cls loss
        # weight = weight.view(-1)
        # target = target.view(-1)
        # preds = preds.view(-1, preds_shape[-1])

        loss = module(preds.view(-1, preds_shape[-1]),
                      target.view(-1)) * weight.view(-1)
        loss = loss.view(batch_size, -1)
    else:
        raise ValueError('can not assume any possible loss type')

    if normalize:
        num_valid = (weight > 0).float().sum(dim=-1).clamp(min=1)
        return loss.sum(dim=-1) / num_valid
    else:
        return loss.sum(dim=-1)


# CrossEntropyLoss = WeightedLossWrapper(nn.CrossEntropyLoss)
# SmoothL1Loss = WeightedLossWrapper(nn.SmoothL1Loss)
