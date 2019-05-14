# -*- coding: utf-8 -*-

import torch.nn as nn
import torch


class CornersLoss(nn.Module):
    def __init__(self, split_loss=False, use_filter=True):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
        self.reg_loss = nn.MSELoss(reduction='none')
        self.split_loss = split_loss
        self.use_filter = use_filter

    def forward(self, preds, targets):
        """
        Args:
            preds: shape(N, )
            targets: shape(N, )
        Returns:
            pass
        """
        N, M = preds.shape[:2]
        encoded_corners_2d_all = preds.view(N, M, 8, 4)
        corners_preds = encoded_corners_2d_all[:, :, :, :2].contiguous().view(
            N, M, -1)
        cls_preds = encoded_corners_2d_all[:, :, :, 2:].contiguous().view(N, M,
                                                                          -1)

        targets = targets.view(N, M, 8, 3)
        corners_gt = targets[:, :, :, :2].contiguous().view(N, M, -1)
        cls_gt = targets[:, :, :, 2:].contiguous().view(N, M, -1).long()

        cls_loss = self.cls_loss(cls_preds.contiguous().view(-1, 2),
                                 cls_gt.contiguous().view(-1))
        reg_loss = self.reg_loss(corners_preds, corners_gt)

        batch_size = reg_loss.shape[0]
        num_samples = reg_loss.shape[1]
        # import ipdb
        # ipdb.set_trace()
        if self.use_filter:
            reg_loss = reg_loss.view(-1, 2) * cls_gt.view(-1, 1).float()
        else:
            reg_loss = reg_loss.view(-1, 2)
        #  reg_loss = reg_loss.view(-1, 2)
        total_loss = torch.cat([cls_loss.unsqueeze(-1), reg_loss], dim=-1)
        return total_loss.view(batch_size, num_samples, -1)
