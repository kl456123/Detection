# -*- coding: utf-8 -*-

import torch.nn as nn
import torch


class CornersLoss(nn.Module):
    def __init__(self, split_loss=False, use_filter=True,
                 training_depth=False):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
        self.reg_loss = nn.MSELoss(reduction='none')
        self.split_loss = split_loss
        self.use_filter = use_filter
        self.training_depth = training_depth

    def forward(self, preds, targets):
        """
        Args:
            preds: shape(N, )
            targets: shape(N, )
        Returns:
            pass
        """
        if self.training_depth:
            num_reg_col = 3
        else:
            num_reg_col = 2

        N, M = preds.shape[:2]
        encoded_corners_2d_all = preds.view(N, M, 8, 5)
        corners_preds = encoded_corners_2d_all[:, :, :, :3].contiguous().view(
            N, M, -1)
        cls_preds = encoded_corners_2d_all[:, :, :, 3:].contiguous().view(
            N, M, -1)

        targets = targets.view(N, M, 8, 4)
        corners_gt = targets[:, :, :, :3].contiguous().view(N, M, -1)
        cls_gt_weight = targets[:, :, :, 3:].contiguous().view(N, M, -1)
        cls_gt = (cls_gt_weight > 0).long()

        cls_loss = self.cls_loss(cls_preds.contiguous().view(-1, 2),
                                 cls_gt.contiguous().view(-1))

        corners_preds = corners_preds.view(N, M, 8, 3)
        corners_gt = corners_gt.view(N, M, 8, 3)
        corners_loss = self.reg_loss(
            corners_preds[:, :, :, :2].contiguous().view(N, M, -1),
            corners_gt[:, :, :, :2].contiguous().view(N, M, -1))

        if self.training_depth:
            depth_loss = self.reg_loss(
                corners_preds[:, :, :, 2:].contiguous().view(N, M, -1),
                corners_gt[:, :, :, 2:].contiguous().view(N, M, -1))
            reg_loss = torch.cat([corners_loss, depth_loss], dim=-1)
        else:
            reg_loss = corners_loss

        batch_size = reg_loss.shape[0]
        num_samples = reg_loss.shape[1]
        # import ipdb
        # ipdb.set_trace()
        if self.use_filter:
            reg_loss = reg_loss.view(-1, num_reg_col) * cls_gt_weight.view(-1, 1).float()
        else:
            reg_loss = reg_loss.view(-1, num_reg_col)
        #  reg_loss = reg_loss.view(-1, 2)
        total_loss = torch.cat([cls_loss.unsqueeze(-1), reg_loss], dim=-1)
        return total_loss.view(batch_size, num_samples, -1)
