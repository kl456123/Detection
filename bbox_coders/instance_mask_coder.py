# -*- coding: utf-8 -*-
import torch

from utils.registry import BBOX_CODERS
from core import constants
from utils import geometry_utils


@BBOX_CODERS.register(constants.KEY_INSTANCES_MASK)
class InstanceMaskCoder(object):
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
    def encode_batch(proposals, label_instance_mask, match):
        """
        xyxy
        Args:
            anchors: shape(N, M, 4)
            gt_boxes: shape(N, M, 4)
        Returns:
            target: shape(N, M, 4)
        """
        N, M = proposals.shape[:2]
        H, W = label_instance_mask.shape[-2:]
        proposals = proposals.long()
        depth_targets = []
        for batch_ind in range(N):
            proposals_single_image = proposals[batch_ind]
            pos_proposals_single_image = proposals_single_image[
                weights[batch_ind] > 0]
            for proposals_ind in range(pos_proposals_single_image.shape[0]):
                box = pos_proposals_single_image[proposals_ind]
                depth_targets.append(
                    F.upsample_bilinear(
                        depth_map[batch_ind:batch_ind + 1, :, box[1]:box[3],
                                  box[0]:box[2]],
                        size=mask_size))

        depth_targets = torch.cat(depth_targets, dim=1)
        return depth_targets.view(-1, mask_size * mask_size)
