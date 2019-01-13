# -*- coding: utf-8 -*-

import torch.nn as nn


class OrientationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss(reduce=False)
        self.reg_loss = nn.MSELoss(reduce=False)

    def forward(self, preds, targets):
        """
        Args:
            preds: shape()
            targets: shape(N, 3), (cls_orient(1), reg_orient(2),h_2d(1),c_2d(2))
        """
        # import ipdb
        # ipdb.set_trace()
        # cls loss
        cls_orient = targets[:, 0].long()
        cls_preds = preds[:, :2]

        cls_loss = self.cls_loss(cls_preds, cls_orient)

        # reg loss
        reg_orient = targets[:, 1:]
        reg_preds = preds[:, 2:]
        reg_loss = self.reg_loss(reg_preds, reg_orient)
        orient_loss = reg_loss.sum(dim=-1) + cls_loss
        return orient_loss
