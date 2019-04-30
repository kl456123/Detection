# -*- coding: utf-8 -*-

"""
Bugs here !!!
Don't use it for training
"""

import torch.nn as nn
import torch
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='none'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, x, y, size_average=False):
        """Focal loss

        Args:
            x(tensor): size [N, D]
            y(tensor): size [N, ]
            alpha(float)
            gamma(float)
            size_average
        Returns:
            (tensor): focal loss
        """
        alpha = self.alpha
        gamma = self.gamma
        with torch.no_grad():
            alpha_t = torch.ones(x.size()) * alpha
            alpha_t[:, 0] = 1 - alpha
            alpha_t = alpha_t.cuda().gather(1, y.view(-1, 1))
        pt = F.softmax(x, dim=1).gather(1, y.view(-1, 1))
        _loss = -alpha_t * torch.log(pt) * torch.pow((1 - pt), gamma)

        if self.reduction == 'none':
            return _loss.sum(dim=-1)
        else:
            return torch.sum(_loss)
