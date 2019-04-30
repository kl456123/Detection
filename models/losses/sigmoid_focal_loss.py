# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


def sigmoid_focal_loss_cpu(logits, targets, gamma, alpha):
    num_classes = logits.shape[1]
    gamma = gamma
    alpha = alpha
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(
        1, num_classes + 1, dtype=dtype, device=device).unsqueeze(0)

    t = targets.unsqueeze(1)
    p = torch.sigmoid(logits)
    term1 = (1 - p)**gamma * torch.log(p)
    term2 = p**gamma * torch.log(1 - p)
    return -(t == class_range).float() * term1 * alpha - (
        (t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma, alpha):
        super(SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        # if logits.is_cuda:
        # loss_func = sigmoid_focal_loss_cuda
        # else:
        loss_func = sigmoid_focal_loss_cpu

        loss = loss_func(logits, targets, self.gamma, self.alpha)
        return loss.sum(dim=-1)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        tmpstr += ")"
        return tmpstr
