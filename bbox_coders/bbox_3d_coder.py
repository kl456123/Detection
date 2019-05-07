# -*- coding: utf-8 -*-
import torch
from torch.nn import functional as F
from utils.registry import BBOX_CODERS


@BBOX_CODERS.register('bbox_3d')
class BBox3DCoder(object):
    mean_dims = None

    @staticmethod
    def encode_batch_bbox(dims, proposal_bboxes, assigned_gt_labels):
        """
        encoding dims may be better,here just encode dims_2d
        Args:
            dims: shape(N,6), (h,w,l) and their projection in 2d
        """
        device = dims.device
        mean_dims = BBox3DCoder.mean_dims.to(device)

        #  h_3d_mean = 1.67
        #  w_3d_mean = 1.87
        #  l_3d_mean = 3.7

        #  h_3d_std = 1
        #  w_3d_std = 1
        #  l_3d_std = 1

        #  target_h_3d = (dims[:, 0] - h_3d_mean) / h_3d_std
        #  target_w_3d = (dims[:, 1] - w_3d_mean) / w_3d_std
        #  target_l_3d = (dims[:, 2] - l_3d_mean) / l_3d_std
        #  targets = torch.stack([target_h_3d, target_w_3d, target_l_3d], dim=-1)
        #  import ipdb
        #  ipdb.set_trace()
        bg_mean_dims = torch.zeros_like(mean_dims[:, -1:, :])
        mean_dims = torch.cat([bg_mean_dims, mean_dims], dim=1)
        assigned_mean_dims = mean_dims[0][assigned_gt_labels].float()
        assigned_std_dims = torch.ones_like(assigned_mean_dims)
        targets = (dims[:, :3] - assigned_mean_dims) / assigned_std_dims

        #  keypoint_gt = self.encode_batch_keypoint(dims[:, 3:], num_intervals,
        #  rois_batch)
        reg_orient = dims[:, 4:6]
        # normalize it using rois_batch
        # w = proposal_bboxes[:, 2] - proposal_bboxes[:, 0] + 1
        # h = proposal_bboxes[:, 3] - proposal_bboxes[:, 1] + 1
        #  reg_orient[:, 0] = reg_orient[:, 0] / w
        #  reg_orient[:, 1] = reg_orient[:, 1] / h

        targets = torch.cat([targets, dims[:, 3:4], reg_orient], dim=-1)
        return targets

    @staticmethod
    def decode_batch_bbox(targets, rois):
        device = rois.device
        mean_dims = BBox3DCoder.mean_dims.to(device)

        # dims
        #  h_3d_mean = 1.67
        #  w_3d_mean = 1.87
        #  l_3d_mean = 3.7

        #  h_3d_std = 1
        #  w_3d_std = 1
        #  l_3d_std = 1

        #  h_3d = targets[:, 0] * h_3d_std + h_3d_mean
        #  w_3d = targets[:, 1] * w_3d_std + w_3d_mean
        #  l_3d = targets[:, 2] * l_3d_std + l_3d_mean
        bg_mean_dims = torch.zeros_like(mean_dims[:, -1:, :])
        mean_dims = torch.cat([bg_mean_dims, mean_dims], dim=1).float()
        # assigned_mean_dims = mean_dims[0][pred_labels].float()
        std_dims = torch.ones_like(mean_dims)
        #  targets = (dims[:, :3] - assigned_mean_dims) / assigned_std_dims
        bbox = targets[:, :-5].view(targets.shape[0], -1,
                                    3) * std_dims + mean_dims
        bbox = bbox.view(targets.shape[0], -1)
        #  bbox = torch.stack([h_3d, w_3d, l_3d], dim=-1)

        # rois w and h

        # cls orient
        cls_orient = targets[:, 3:6]
        cls_orient = F.softmax(cls_orient, dim=-1)
        cls_orient, cls_orient_argmax = torch.max(cls_orient, dim=-1)

        reg_orient = targets[:, 6:8]

        #  reg_orient[:, 0] = reg_orient[:, 0] * w
        #  reg_orient[:, 1] = reg_orient[:, 1] * h

        orient = torch.stack(
            [
                cls_orient_argmax.type_as(reg_orient), reg_orient[:, 0],
                reg_orient[:, 1]
            ],
            dim=-1)

        return torch.cat([bbox, orient], dim=-1)
