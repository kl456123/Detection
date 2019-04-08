# -*- coding: utf-8 -*-

from utils.registry import BBOX_CODERS
from core import constants
from utils import geometry_utils
import torch
import torch.nn.functional as F


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
    def decode_batch(orients, rpn_proposals, rcnn_proposals, p2):
        """
        Args:
            orients: shape(N, )
        """

        assert orients.shape[-1] == 4
        cls_orients = orients[:, :, :2]
        reg_orients = orients[:, :, 2:]
        cls_orients = F.softmax(cls_orients, dim=-1)
        _, cls_orients_argmax = torch.max(cls_orients, keepdim=True, dim=-1)

        rpn_proposals_xywh = geometry_utils.torch_xyxy_to_xywh(rpn_proposals)
        reg_orients = reg_orients * rpn_proposals_xywh[:, :, 2:]

        orients = torch.cat(
            [cls_orients_argmax.type_as(reg_orients), reg_orients], dim=-1)

        side_points = OrientsCoder._generate_side_points(rcnn_proposals,
                                                         orients)

        ry = geometry_utils.torch_pts_2d_to_dir_3d(side_points, p2)

        return ry

    @staticmethod
    def _decode_reg_orients(reg_orient):
        pass

    @staticmethod
    def _generate_side_points(dets_2d, orient):
        """
        Generate side points to calculate orientation
        Args:
            dets_2d: shape(N,4) detected box
            orient: shape(N,3) cls_orient and reg_orient
        Returns:
            side_points: shape(N,4)
        """
        assert orient.shape[-1] == 3
        cls_orient_argmax = orient[:, :, 0].long()

        reg_orient = orient[:, :, 1:3]

        side_points = torch.zeros(
            (orient.shape[0], orient.shape[1], 4)).type_as(orient)

        # cls_orient_argmax = np.argmax(cls_orient, axis=-1)

        row_inds = torch.arange(0, cls_orient_argmax.shape[1]).long()

        # two points
        selected_x = torch.stack([dets_2d[:, :, 2], dets_2d[:, :, 0]], dim=-1)
        # side point
        side_points[:, :, 3] = dets_2d[:, :, 3] - reg_orient[:, :, 1]

        side_points[:, :, 2] = selected_x.view(
            -1, 2)[row_inds.view(-1), cls_orient_argmax.view(-1)].view_as(
                cls_orient_argmax)

        # bottom point
        selected_x = torch.stack(
            [
                dets_2d[:, :, 2] - reg_orient[:, :, 0],
                dets_2d[:, :, 0] + reg_orient[:, :, 0]
            ],
            dim=-1)
        side_points[:, :, 1] = dets_2d[:, :, 3]
        side_points[:, :, 0] = selected_x.view(
            -1, 2)[row_inds.view(-1), cls_orient_argmax.view(-1)].view_as(
                cls_orient_argmax)
        return side_points

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
