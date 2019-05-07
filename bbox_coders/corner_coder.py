# -*- coding: utf-8 -*-
import torch

from utils.registry import BBOX_CODERS
from core import constants
from utils import geometry_utils


@BBOX_CODERS.register('corner')
class CornerCoder(object):
    @staticmethod
    def decode_batch(deltas, anchors):
        """
        Args:
            deltas: shape(N, M, 4)
            boxes: shape(N, M, 4)
        """
        variances = [0.1, 0.2]
        anchors_xywh = geometry_utils.torch_xyxy_to_xywh(anchors)
        wh = anchors_xywh[:, :, 2:]
        xymin = anchors[:, :, :2] + deltas[:, :, :2] * wh * variances[0]
        xymax = anchors[:, :, 2:] + deltas[:, :, 2:] * wh * variances[0]
        return torch.cat([xymin, xymax], dim=-1)

    @staticmethod
    def encode_batch(anchors, gt_boxes):
        """
        xyxy
        Args:
            anchors: shape(N, M, 4)
            gt_boxes: shape(N, M, 4)
        Returns:
            target: shape(N, M, 4)
        """
        variances = [0.1, 0.2]
        anchors_xywh = geometry_utils.torch_xyxy_to_xywh(anchors)
        wh = anchors_xywh[:, :, 2:]
        xymin = (gt_boxes[:, :, :2] - anchors[:, :, :2]) / (variances[0] * wh)
        xymax = (gt_boxes[:, :, 2:] - anchors[:, :, 2:]) / (variances[0] * wh)
        return torch.cat([xymin, xymax], dim=-1)
