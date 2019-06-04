# -*- coding: utf-8 -*-

import torch.nn as nn
import torch


class CornersStableLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.cls_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, preds, targets):
        """
        Args:
            preds: shape(N, M, 8*(4*2 + 4))
            targets: shape(N, M, 8*(4*(2+1) + 1))
        Returns:
            pass
        """

        # import ipdb
        # ipdb.set_trace()
        N, M = preds.shape[:2]

        preds = preds.view(N, M, -1)
        targets = targets.view(N, M, -1)

        # reg loss
        reg_preds = preds[..., :64]
        reg_targets = targets[..., :64]
        visibility = targets[..., 64:64 + 32]
        # reg_preds = reg_preds.view(N, M, 8, 4, -1)
        # reg_targets = reg_targets.view(N, M, 8, 4, -1)

        # weights = targets[..., 8:12]
        reg_loss = self.mse(reg_preds.contiguous().view(-1, 2),
                            reg_targets.contiguous().view(-1, 2)
                            ) * visibility.contiguous().view(-1).unsqueeze(-1)

        # cls loss
        cls_preds = preds[..., 64:]
        cls_targets = targets[..., 64 + 32:].long()
        cls_loss = self.cls_loss(cls_preds.contiguous().view(-1, 4),
                                 cls_targets.contiguous().view(-1))

        total_loss = torch.cat(
            [reg_loss.view(N, M, -1),
             cls_loss.view(N, M, -1)], dim=-1)

        # preds = preds.view(N, M, 8, 4, -1)
        # targets = targets.view(N, M, 8, 4, -1)

        # reg_targets = targets[..., :2]
        # weights = targets[..., -2:-1]
        # cls_targets = targets[..., -1:].long()
        # reg_preds = preds[..., :2]
        # corners_loss = self.mse(reg_preds, reg_targets) * weights

        # # cls loss
        # cls_preds = preds[..., 2:]
        # cls_loss = self.cls_loss(cls_preds.view(-1, 4), cls_targets.view(-1))

        # total_loss = torch.cat(
        # [corners_loss, cls_loss.view(N, M, 8, 4, 1)], dim=-1)

        return total_loss.view(N, M, -1)
