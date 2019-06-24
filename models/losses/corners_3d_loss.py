# -*- coding: utf-8 -*-

import torch.nn as nn
import torch


class Corners3DLoss(nn.Module):
    def __init__(self, split_loss=False, use_filter=True):
        super().__init__()
        self.l1_loss = nn.SmoothL1Loss(reduction='none')
        self.split_loss = split_loss

    def forward(self, preds, targets):
        """
        Args:
            preds: shape(N, )
            targets: shape(N, )
        Returns:
            pass
        """
        # N, M = preds.shape[:2]

        local_corners_3d_preds = preds[:, :, :24]
        local_corners_3d_gt = targets[:, :, :24]

        C_2d_preds = preds[:, :, 24:26]
        C_2d_gt = targets[:, :, 24:26]

        instance_depth_preds = preds[:, :, 26:]
        instance_depth_gt = targets[:, :, 26:]

        local_corners_3d_loss = self.l1_loss(local_corners_3d_preds,
                                             local_corners_3d_gt)
        C_2d_loss = self.l1_loss(C_2d_preds, C_2d_gt)
        instance_depth_loss = self.l1_loss(instance_depth_preds,
                                           instance_depth_gt)

        loss = torch.cat(
            [local_corners_3d_loss * 10, C_2d_loss, instance_depth_loss],
            dim=-1)

        # return self.l1_loss(preds, targets)
        return loss
