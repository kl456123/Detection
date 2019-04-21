# -*- coding: utf-8 -*-

import torch.nn as nn
import torch


class OrientationLoss(nn.Module):
    def __init__(self, split_loss=False):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
        self.reg_loss = nn.MSELoss(reduction='none')
        self.split_loss = split_loss

    def forward(self, preds, targets):
        """
        Args:
            preds: shape()
            targets: shape(N, 3), (cls_orient(1), reg_orient(2),h_2d(1),c_2d(2))
        """

        # cls loss
        # import ipdb
        # ipdb.set_trace()
        num_case_orients = preds.shape[-1] - 2
        cls_orient = targets[:, :, 0].long()
        cls_preds = preds[:, :, :num_case_orients]
        cls_loss = self.cls_loss(
            cls_preds.view(-1, num_case_orients),
            cls_orient.view(-1)).view_as(cls_orient)

        # reg loss
        reg_orient = torch.cat([targets[:, :, 1:3]], dim=-1)
        reg_preds = torch.cat([preds[:, :, -2:]], dim=-1)
        reg_loss = self.reg_loss(reg_preds, reg_orient)
        orinet_loss = torch.cat([cls_loss.unsqueeze(-1), reg_loss], dim=-1)
        return orinet_loss
