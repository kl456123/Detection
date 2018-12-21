# -*- coding: utf-8 -*-
import torch
import math
from core.ops import get_angle


class BBox3DCoder(object):
    def __init__(self, coder_config):
        pass

    def decode_batch(self, deltas, boxes):
        """
        Args:
            deltas: shape(N,K*A,4)
            boxes: shape(N,K*A,4)
        """
        pass
        # if boxes.dim() == 3:

    # pass
    # elif boxes.dim() == 2:
    # boxes = boxes.expand_as(deltas)
    # else:
    # raise ValueError("The dimension of boxes should be 3 or 2")
    # widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    # heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    # ctr_x = boxes[:, :, 0] + 0.5 * widths
    # ctr_y = boxes[:, :, 1] + 0.5 * heights

    # dx = deltas[:, :, 0::4]
    # dy = deltas[:, :, 1::4]
    # dw = deltas[:, :, 2::4]
    # dh = deltas[:, :, 3::4]

    # pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    # pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    # pred_w = torch.exp(dw) * widths.unsqueeze(2)
    # pred_h = torch.exp(dh) * heights.unsqueeze(2)

    # pred_boxes = deltas.clone()
    # # x1
    # pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    # # y1
    # pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # # x2
    # pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # # y2
    # pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    # return pred_boxes

    # def encode_batch(self, bboxes, assigned_gt_boxes):
    # reg_targets_batch = self._encode_batch(bboxes, assigned_gt_boxes)

    # return reg_targets_batch

    def encode_batch(self, boxes_2d, coords):
        """
        Note that bbox_3d is just some points in image about 3d bbox
        Args:
            bbox_2d: shape(N,4)
            bbox_3d: shape(N,7)
        """
        center_x = (boxes_2d[:, 2] + boxes_2d[:, 0]) / 2
        center_y = (boxes_2d[:, 3] + boxes_2d[:, 1]) / 2
        center = torch.stack([center_x, center_y], dim=-1)
        w = (boxes_2d[:, 2] - boxes_2d[:, 0] + 1)
        h = (boxes_2d[:, 3] - boxes_2d[:, 1] + 1)
        dims = torch.stack([w, h], dim=-1)

        bbox_3d = coords[:, :-1].view(-1, 3, 2)
        bbox_3d = (bbox_3d - center.unsqueeze(1)) / dims.unsqueeze(1)
        y = (coords[:, -1:] - center[:, 1:]) / dims[:, 1:]
        coords = torch.cat([bbox_3d.view(-1, 6), y], dim=-1)

        # targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh),
        # 2)

        return coords

    def encode_batch_dims(self, boxes_2d, dims):
        """
        encoding dims may be better,here just encode dims_2d
        Args:
            boxes_2d: shape(N,)
            dims: shape(N,6), (h,w,l) and their projection in 2d
        """
        w = (boxes_2d[:, 2] - boxes_2d[:, 0] + 1)
        h = (boxes_2d[:, 3] - boxes_2d[:, 1] + 1)

        target_h = torch.log(dims[:, 0] / h)
        target_w = torch.log(dims[:, 1] / w)
        target_l = torch.log(dims[:, 2] / w)

        h_3d_mean = 1.67
        w_3d_mean = 1.87
        l_3d_mean = 3.7

        h_3d_std = 1
        w_3d_std = 1
        l_3d_std = 1

        target_h_3d = (dims[:, 3] - h_3d_mean) / h_3d_std
        target_w_3d = (dims[:, 4] - w_3d_mean) / w_3d_std
        target_l_3d = (dims[:, 5] - l_3d_mean) / l_3d_std
        targets = torch.stack(
            [
                target_h, target_w, target_l, target_h_3d, target_w_3d,
                target_l_3d
            ],
            dim=-1)
        return targets

    def encode_batch_bbox(self, dims):
        """
        encoding dims may be better,here just encode dims_2d
        Args:
            boxes_2d: shape(N,)
            dims: shape(N,6), (h,w,l) and their projection in 2d
        """

        h_3d_mean = 1.67
        w_3d_mean = 1.87
        l_3d_mean = 3.7

        h_3d_std = 1
        w_3d_std = 1
        l_3d_std = 1

        target_h_3d = (dims[:, 0] - h_3d_mean) / h_3d_std
        target_w_3d = (dims[:, 1] - w_3d_mean) / w_3d_std
        target_l_3d = (dims[:, 2] - l_3d_mean) / l_3d_std
        targets = torch.stack(
            [target_h_3d, target_w_3d, target_l_3d, dims[:, -1]], dim=-1)
        return targets

    def decode_batch_dims(self, boxes_2d, targets):
        """
        Args:
            boxes_2d: shape(N,)
            targets: shape(N,)
        """
        w = (boxes_2d[:, 2] - boxes_2d[:, 0] + 1)
        h = (boxes_2d[:, 3] - boxes_2d[:, 1] + 1)
        h_2d = h * torch.exp(targets[:, 0])
        w_2d = w * torch.exp(targets[:, 1])
        l_2d = w * torch.exp(targets[:, 2])

        h_3d_mean = 1.67
        w_3d_mean = 1.87
        l_3d_mean = 3.7

        h_3d_std = 1
        w_3d_std = 1
        l_3d_std = 1

        h_3d = targets[:, 3] * h_3d_std + h_3d_mean
        w_3d = targets[:, 4] * w_3d_std + w_3d_mean
        l_3d = targets[:, 5] * l_3d_std + l_3d_mean

        dims = torch.stack([h_2d, w_2d, l_2d, h_3d, w_3d, l_3d], dim=-1)
        return dims

    def decode_batch_bbox(self, targets, bin_centers=None):

        # dims
        h_3d_mean = 1.67
        w_3d_mean = 1.87
        l_3d_mean = 3.7

        h_3d_std = 1
        w_3d_std = 1
        l_3d_std = 1

        h_3d = targets[:, 0] * h_3d_std + h_3d_mean
        w_3d = targets[:, 1] * w_3d_std + w_3d_mean
        l_3d = targets[:, 2] * l_3d_std + l_3d_mean

        # ry
        sin = targets[:, -2]
        cos = targets[:, -1]
        theta = get_angle(sin, cos)
        if bin_centers is not None:
            # theta = bin_centers + theta
            theta = bin_centers

        cond_pos = (cos < 0) & (sin > 0)
        cond_neg = (cos < 0) & (sin < 0)
        theta[cond_pos] = math.pi - theta[cond_pos]
        theta[cond_neg] = -math.pi - theta[cond_neg]

        # ry = torch.atan(sin / cos)
        # cond = cos < 0
        # cond_pos = sin > 0
        # cond_neg = sin < 0
        # ry[cond & cond_pos] = ry[cond & cond_pos] + math.pi
        # ry[cond & cond_neg] = ry[cond & cond_neg] - math.pi

        bbox = torch.stack([h_3d, w_3d, l_3d, theta], dim=-1)
        return bbox
