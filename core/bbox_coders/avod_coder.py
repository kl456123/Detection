# -*- coding: utf-8 -*-

import torch
import numpy as np


class AVODBBoxCoder(object):
    def __init__(self, coder_config):
        pass

    def decode_batch(self, offsets, anchors):
        try:
            x_pred = (offsets[:, 0] * anchors[:, 3]) + anchors[:, 0]
        except:
            import ipdb
            ipdb.set_trace()
        y_pred = (offsets[:, 1] * anchors[:, 4]) + anchors[:, 1]
        z_pred = (offsets[:, 2] * anchors[:, 5]) + anchors[:, 2]

        ry = anchors[:, 6]

        tensor_format = isinstance(anchors, torch.Tensor)
        if tensor_format:
            dx_pred = anchors[:, 3] * torch.exp(offsets[:, 3])
            dy_pred = anchors[:, 4] * torch.exp(offsets[:, 4])
            dz_pred = anchors[:, 5] * torch.exp(offsets[:, 5])
            anchors = torch.stack(
                [x_pred, y_pred, z_pred, dx_pred, dy_pred, dz_pred, ry],
                dim=-1)
        else:
            dx_pred = anchors[:, 3] * np.exp(offsets[:, 3])
            dy_pred = anchors[:, 4] * np.exp(offsets[:, 4])
            dz_pred = anchors[:, 5] * np.exp(offsets[:, 5])
            anchors = np.stack(
                [x_pred, y_pred, z_pred, dx_pred, dy_pred, dz_pred, ry],
                axis=-1)

        return anchors

    def encode_batch(self, anchors, gt_boxes):
        x = torch.cos(gt_boxes[:, :, -1])
        y = torch.sin(gt_boxes[:, :, -1])

        x_target = (gt_boxes[:, :, 0] - anchors[:, :, 0]) / anchors[:, :, 3]
        y_target = (gt_boxes[:, :, 1] - anchors[:, :, 1]) / anchors[:, :, 4]
        z_target = (gt_boxes[:, :, 2] - anchors[:, :, 2]) / anchors[:, :, 5]

        tensor_format = isinstance(anchors, torch.Tensor)
        if tensor_format:
            dx_target = torch.log(gt_boxes[:, :, 3] / anchors[:, :, 3])
            dy_target = torch.log(gt_boxes[:, :, 4] / anchors[:, :, 4])
            dz_target = torch.log(gt_boxes[:, :, 5] / anchors[:, :, 5])
            targets = torch.stack(
                [
                    x_target, y_target, z_target, dx_target, dy_target,
                    dz_target, x, y
                ],
                dim=-1)
        else:
            dx_target = np.log(gt_boxes[:, :, 3] / anchors[:, :, 3])
            dy_target = np.log(gt_boxes[:, :, 4] / anchors[:, :, 4])
            dz_target = np.log(gt_boxes[:, :, 5] / anchors[:, :, 5])
            targets = np.stack(
                [
                    x_target, y_target, z_target, dx_target, dy_target,
                    dz_target, x, y
                ],
                dim=-1)

        return targets
