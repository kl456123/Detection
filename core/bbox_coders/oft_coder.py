# -*- coding: utf-8 -*-
import torch
import math
from core.ops import get_angle
from torch.nn import functional as F


class OFTBBoxCoder(object):
    def __init__(self, coder_config):
        self.etha = coder_config['etha']
        self.dim_mean = coder_config['dim_mean']

    def encode_batch_bbox(self, voxel_centers, gt_boxes_3d):
        """
        encoding dims may be better,here just encode dims_2d
        Args:
            voxel_centers: shape(M,3)
            gt_boxes_3d: shape(N,M,7), (dim,pos,ry)
        """

        # encode dim
        h_3d_mean, w_3d_mean, l_3d_mean = self.dim_mean

        target_h_3d = torch.log(gt_boxes_3d[:, :, 0] / h_3d_mean)
        target_w_3d = torch.log(gt_boxes_3d[:, :, 1] / w_3d_mean)
        target_l_3d = torch.log(gt_boxes_3d[:, :, 2] / l_3d_mean)
        targets_dim = torch.stack(
            [target_h_3d, target_w_3d, target_l_3d], dim=-1)

        # encode pos
        targets_pos = (gt_boxes_3d[:, :, 3:6] - voxel_centers) / self.etha

        # encode angle
        ry = gt_boxes_3d[:, :, -1]
        targets_angle = torch.stack([torch.cos(ry), torch.sin(ry)], dim=-1)

        targets = torch.cat([targets_dim, targets_pos, targets_angle], dim=-1)
        return targets

    def decode_batch_bbox(self, voxel_centers, targets):

        # decode dim
        h_3d_mean, w_3d_mean, l_3d_mean = self.dim_mean
        targets_dim = targets[:, :, :3]
        h_3d = torch.exp(targets_dim[:, :, 0]) * h_3d_mean
        w_3d = torch.exp(targets_dim[:, :, 1]) * w_3d_mean
        l_3d = torch.exp(targets_dim[:, :, 2]) * l_3d_mean

        decoded_dims = torch.stack([h_3d, w_3d, l_3d], dim=-1)

        # decode pos
        targets_pos = targets[:, :, 3:6]
        decoded_pos = voxel_centers + targets_pos * self.etha

        # decode angle
        targets_angle = targets[:, :, 6:]
        ry = -torch.atan2(targets_angle[:, :, 1],
                          targets_angle[:, :, 0]).unsqueeze(-1)

        return torch.cat([decoded_dims, decoded_pos, ry], dim=-1)

    def encode_batch_labels(self, voxel_centers, gt_boxes_3d):
        """
        Args:
            voxel_centers: shape(N, 3)
            gt_labels: shape(num_batch, M)
            gt_boxes_3d: shape(num_batch, M, 7)
        """
        pos = gt_boxes_3d[:, :, 3:6]

        pos = pos.unsqueeze(2)

        gt_x = pos[:, :, :, 0]
        gt_z = pos[:, :, :, 2]

        voxel_centers = voxel_centers.unsqueeze(0).unsqueeze(0)

        voxel_x = voxel_centers[:, :, :, 0]
        voxel_z = voxel_centers[:, :, :, 2]

        # shape(num_batch, M, N)
        scores_map = torch.exp(-(torch.pow((gt_x - voxel_x), 2) + torch.pow(
            (gt_z - voxel_z), 2)) / (2 * self.etha * self.etha))
        scores_map = scores_map.max(dim=1)[0]
        return scores_map
