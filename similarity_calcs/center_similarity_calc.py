#!/usr/bin/env python
# encoding: utf-8

import torch
from utils.registry import SIMILARITY_CALCS
from core.similarity_calc import SimilarityCalc


@SIMILARITY_CALCS.register('center')
class CenterSimilarityCalc(SimilarityCalc):
    def compare_batch(self, anchors, gt_boxes):
        """
        Args:
            anchors: shape(N, M, 4)
            gt_boxes: shape(N, K, 4)
        Returns:
            overlaps: shape(N, M, K)
        """
        lt = torch.max(anchors[:, :, None, :2], gt_boxes[:, None, :, :2])
        rb = torch.min(anchors[:, :, None, 2:], gt_boxes[:, None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, :, 0] * wh[:, :, :, 1]

        area1 = (anchors[:, :, 2] - anchors[:, :, 0]) * (
            anchors[:, :, 3] - anchors[:, :, 1])
        area2 = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0]) * (
            gt_boxes[:, :, 3] - gt_boxes[:, :, 1])

        iou = inter / (area1[:, :, None] + area2[:, None] - inter)

        return iou
