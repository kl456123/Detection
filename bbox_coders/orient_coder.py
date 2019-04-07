# -*- coding: utf-8 -*-

from utils.registry import BBOX_CODERS
from core import constants
from utils import geometry_utils
import torch


@BBOX_CODERS.register(constants.KEY_ORIENTS)
class OrientsCoder(object):
    @staticmethod
    def encode(label_boxes_3d, proposals, p2):
        label_corners_2d = geometry_utils.torch_boxes_3d_to_corners_2d(
            label_boxes_3d, p2)

        # shape(N, 2, 2)
        center_side = OrientsCoder._get_center_side(label_corners_2d)

        # label_boxes_2d_proj = geometry_utils.corners_2d_to_boxes_2d(
        # label_corners_2d)

        label_orients = OrientsCoder._generate_orients(center_side, proposals)
        return label_orients

    @staticmethod
    def encode_batch(label_boxes_3d, proposals, p2):
        batch_size = label_boxes_3d.shape[0]
        orients_batch = []
        for batch_ind in range(batch_size):
            orients_batch.append(
                OrientsCoder.encode(label_boxes_3d[batch_ind],
                                    proposals[batch_ind], p2[batch_ind]))
        return torch.stack(orients_batch, dim=0)

    @staticmethod
    def decode_batch(orients, p2):
        pass

    @staticmethod
    def _decode_reg_orients(reg_orient):
        pass

    @staticmethod
    def _generate_orients(center_side, proposals):
        """
        Args:
            boxes_2d_proj: shape(N, 4)
            center_side: shape(N, 2, 2)
        """
        direction = center_side[:, 0] - center_side[:, 1]
        cond = (direction[:, 0] * direction[:, 1]) == 0
        cls_orients = torch.zeros_like(cond).float()
        cls_orients[cond] = -1
        cls_orients[~cond] = (
            (direction[~cond, 1] / direction[~cond, 0]) > 0).float()

        reg_orients = torch.abs(direction)
        proposals_xywh = geometry_utils.torch_xyxy_to_xywh(
            proposals.unsqueeze(0)).squeeze(0)
        reg_orients = reg_orients / proposals_xywh[:, 2:]
        # encode

        return torch.cat([cls_orients.unsqueeze(-1), reg_orients], dim=-1)

    @staticmethod
    def _get_center_side(corners_xy):
        """
        Args:
            corners_xy: shape(N, 8, 2)
        """
        point0 = corners_xy[:, 0]
        point1 = corners_xy[:, 1]
        point2 = corners_xy[:, 2]
        point3 = corners_xy[:, 3]
        mid0 = (point0 + point1) / 2
        mid1 = (point2 + point3) / 2
        return torch.stack([mid0, mid1], dim=1)
