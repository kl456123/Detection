# -*- coding: utf-8 -*-
import torch

from utils.registry import BBOX_CODERS
from core import constants


# @BBOX_CODERS.register(constants.KEY_BOXES_2D_CENTER)
class CenterCoder(object):
    @staticmethod
    def decode_batch(deltas, boxes):
        """
        Args:
            deltas: shape(N,K*A,4)
            boxes: shape(N,K*A,4)
        """
        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0]
        dy = deltas[:, :, 1]
        dw = deltas[:, :, 2]
        dh = deltas[:, :, 3]

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes = deltas.clone()
        # x1
        pred_boxes[:, :, 0] = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes[:, :, 1] = pred_ctr_y - 0.5 * pred_h
        # x2
        pred_boxes[:, :, 2] = pred_ctr_x + 0.5 * pred_w
        # y2
        pred_boxes[:, :, 3] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes

    @staticmethod
    def encode_batch(ex_rois, gt_rois):
        ex_widths = ex_rois[:, :, 2] - ex_rois[:, :, 0]
        ex_heights = ex_rois[:, :, 3] - ex_rois[:, :, 1]
        ex_ctr_x = ex_rois[:, :, 0] + 0.5 * ex_widths
        ex_ctr_y = ex_rois[:, :, 1] + 0.5 * ex_heights

        gt_widths = gt_rois[:, :, 2] - gt_rois[:, :, 0]
        gt_heights = gt_rois[:, :, 3] - gt_rois[:, :, 1]
        gt_ctr_x = gt_rois[:, :, 0] + 0.5 * gt_widths
        gt_ctr_y = gt_rois[:, :, 1] + 0.5 * gt_heights

        targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = torch.log(gt_widths / ex_widths)
        targets_dh = torch.log(gt_heights / ex_heights)

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh),
                              2)
        return targets
