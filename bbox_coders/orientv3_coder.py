# -*- coding: utf-8 -*-
"""
ry is encoded used for MultibinLoss (cls loss and reg loss)
"""
from utils.registry import BBOX_CODERS
from core import constants
from utils import geometry_utils
import torch
import torch.nn.functional as F


@BBOX_CODERS.register(constants.KEY_ORIENTS_V3)
class OrientsV3Coder(object):
    @staticmethod
    def encode_batch(label_boxes_3d):
        # global ry to local ry
        location = label_boxes_3d[:, :, :3]
        ry = label_boxes_3d[:, :, -1]
        alpha = torch.atan2(location[:, :, 2], location[:, :, 0])
        # angle that is clockwise is positive
        ry_local = ry - (-alpha)

        return ry_local.unsqueeze(-1)

    @staticmethod
    def decode_batch(orient_preds, bin_centers):
        """
        Args:
            bin_centers: shape(num_bins)
            orient_preds: shape(N, num, num_bins*4)
        Returns:
            theta: shape(N, num)
        """
        batch_size = orient_preds.shape[0]
        num = orient_preds.shape[1]
        orient_preds = orient_preds.view(batch_size, num, -1, 4)
        num_bins = orient_preds.shape[2]

        angles_cls = F.softmax(orient_preds[:, :, :, :2], dim=-1)
        _, angles_cls_argmax = torch.max(angles_cls[:, :, :, 1], dim=-1)
        row = torch.arange(
            0, angles_cls_argmax.numel()).type_as(angles_cls_argmax)
        angles_oritations = orient_preds[:, :, :, 2:].view(
            -1, num_bins,
            2)[row, angles_cls_argmax.view(-1)].view(batch_size, -1, 2)

        bin_centers = bin_centers[angles_cls_argmax]
        theta = torch.atan2(angles_oritations[:, :, 1],
                            angles_oritations[:, :, 0])
        theta = bin_centers + theta
        return theta
