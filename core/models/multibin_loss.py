# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import math


class MultiBinLoss(nn.Module):
    def __init__(self, num_bins=4, overlaps=1 / 6 * math.pi):
        super().__init__()
        self.num_bins = num_bins
        self.angle_cls_loss = nn.CrossEntropyLoss(reduce=False)
        self.overlaps = overlaps

        # import ipdb
        # ipdb.set_trace()
        interval = 2 * math.pi / self.num_bins

        # bin centers
        bin_centers = torch.arange(0, self.num_bins) * interval
        bin_centers = bin_centers.cuda()
        cond = bin_centers > math.pi
        bin_centers[cond] = bin_centers[cond] - 2 * math.pi

        self.max_deltas = (interval + overlaps) / 2
        # self.left = bin_centers - 1 / 2 * interval
        # self.right = bin_centers + 1 / 2 * interval
        self.bin_centers = bin_centers

    def generate_cls_targets(self, local_angle):
        """
        local_angle ranges from [-pi, pi]
        """
        # cls_targets = (local_angle >= self.left) & (local_angle < self.right)
        deltas = torch.abs(local_angle - self.bin_centers)
        cls_targets = (deltas <= self.max_deltas) | (
            deltas > 2 * math.pi - self.max_deltas)
        return cls_targets.long()

    def forward(self, preds, targets):
        """
        data format of preds: num_bins * (conf, sin, cos)
        data format of targets: local_angle
        Args:
            preds: shape(N,num*4)
            targets: shape(N,1)
        """
        # import ipdb
        # ipdb.set_trace()
        preds = preds.view(-1, self.num_bins, 4)
        # targets[targets < 0] = targets[targets < 0] + 2 * math.pi

        # generate cls target
        cls_targets = self.generate_cls_targets(targets)

        # cls loss
        angle_cls_loss = self.angle_cls_loss(
            preds[:, :, :2].contiguous().view(-1, 2), cls_targets.view(-1))
        angle_cls_loss = angle_cls_loss.view(-1, self.num_bins)
        angle_cls_loss = angle_cls_loss.sum(dim=-1)

        # residual loss
        # reg_targets = self.generate_reg_targets(targets)
        theta = self.get_angle(preds[:, :, 1], preds[:, :, 2])
        angle_reg_weights = cls_targets.detach().float()
        angle_reg_loss = -angle_reg_weights * torch.cos(
            targets - self.bin_centers - theta)
        num_covered = angle_reg_weights.sum(dim=-1)
        angle_reg_loss = 1 / num_covered * angle_reg_loss.sum(dim=-1)

        return angle_cls_loss + angle_reg_loss

    def get_angle(self, sin, cos):
        """
        Args:
            sin: shape(N,num_bins)
        """
        sin = sin.detach()
        cos = cos.detach()
        norm = torch.sqrt(sin * sin + cos * cos)
        sin /= norm
        cos /= norm

        # range in [-pi, pi]
        theta = torch.asin(sin)
        cond_pos = (cos < 0) & (sin > 0)
        cond_neg = (cos < 0) & (sin > 0)
        theta[cond_pos] = math.pi - theta[cond_pos]
        theta[cond_neg] = -math.pi - theta[cond_neg]
        return theta