# -*- coding: utf-8 -*-

import torch.nn as nn
import torch


class KeyPointLoss(nn.Module):
    def __init__(self, reg_refine=False):
        super().__init__()

        self.cls_loss = nn.CrossEntropyLoss(reduction='none')
        self.reg_loss = nn.MSELoss(reduction='none')
        self.reg_refine = reg_refine

    def forward(self, preds, targets):
        """
        Args:
            preds: shape(N, M, 8*3 *56*56)
            targets: shape(N, M, 8*4) (peak_pos(1), peak_offset(2), vis(1))
        """
        # import ipdb
        # ipdb.set_trace()

        N, M = preds.shape[:2]
        hm_preds = preds.view(N, M, 8, -1)
        targets = targets.view(N, M, 8, -1)

        # visibility
        visibility = targets[..., -1]
        hm_targets = targets[..., :-1]

        total_loss = self.reg_loss(hm_preds, hm_targets)
        total_loss = total_loss * visibility.unsqueeze(-1)
        return total_loss.view(N, M, -1)

        # cls loss
        # cls_preds = preds[:, :, :, 0]
        # cls_targets = targets[..., 0].long()
        # cls_loss = self.cls_loss(
        # cls_preds.view(-1, 56 * 56), cls_targets.view(-1))

        # # reg loss
        # reg_preds = preds[:, :, :, 1:]
        # reg_targets = targets[..., 1:3]

        # reg_preds = reg_preds.permute(0, 1, 2, 4, 3).view(N * M * 8, -1, 2)

        # # import ipdb
        # # ipdb.set_trace()
        # # get the peak reg preds
        # # row = torch.arange(cls_targets.numel()).type_as(cls_targets)
        # # peak_reg_preds = reg_preds[row, cls_targets.view(-1)].view(N, M, 8, 2)

        # if self.reg_refine:
        # reg_loss = self.reg_loss(peak_reg_preds, reg_targets)
        # total_loss = torch.cat(
        # [cls_loss.view(N, M, -1, 1), reg_loss], dim=-1)
        # else:
        # total_loss = cls_loss.view(N, M, -1, 1)
        # total_loss = total_loss * visibility.unsqueeze(-1)
        # return total_loss.view(N, M, -1)
