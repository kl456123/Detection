# -*- coding: utf-8 -*-

import torch.nn as nn
import torch


class MultiTaskLoss(nn.Module):
    def __init__(self, num_losses):
        super().__init()

        loss_weights = nn.ParameterList()
        for _ in range(num_losses):
            loss_weights.append(nn.Parameter(torch.Tensor(1)))

        self.num_losses = num_losses
        self.loss_weights = loss_weights
        self.register_parameter()

    def forward(self, losses):
        """
        Args:
            losses: list of loss tensor
        """
        loss = 0
        for i in range(self.num_losses):
            precision = torch.exp(-self.loss_weights[i])
            loss = loss + precision * losses[i] + self.loss_weights[i]

        return loss
