# -*- coding: utf-8 -*-

from utils.registry import BBOX_CODERS
from core import constants
from utils import geometry_utils
import torch
import torch.nn.functional as F
from utils.visualize import visualize_bbox, read_img
from utils.box_vis import draw_line


@BBOX_CODERS.register(constants.KEY_ORIENTS_V2)
class OrientsCoder(object):
    @staticmethod
    def encode(label_boxes_3d, proposals, p2):
        """
        Args:
            label_boxes_3d: shape(N, 7)
            proposals: shape(N, 4)
            p2: shape(3, 4)
        """
        # import ipdb
        # ipdb.set_trace()
        # shape(N, 8, 3)
        corners_3d = geometry_utils.torch_boxes_3d_to_corners_3d(
            label_boxes_3d)
        corners_2d = geometry_utils.torch_points_3d_to_points_2d(
            corners_3d.reshape((-1, 3)), p2).reshape(-1, 8, 2)
        # shape(N, 3)
        left_side_points_3d = (corners_3d[:, 0] + corners_3d[:, 3]) / 2
        right_side_points_3d = (corners_3d[:, 1] + corners_3d[:, 2]) / 2

        # shape(N, 2, 2)
        left_side = torch.stack([corners_2d[:, 0], corners_2d[:, 3]], dim=1)
        right_side = torch.stack([corners_2d[:, 1], corners_2d[:, 2]], dim=1)

        # shape(N, 2, 2, 2)
        side = torch.stack([left_side, right_side], dim=1)

        # no rotation
        K = p2[:3, :3]
        KT = p2[:, -1]
        T = torch.matmul(torch.inverse(K), KT)
        C = -T
        # shape(N, )
        left_dist = torch.norm(left_side_points_3d - C, dim=-1)
        right_dist = torch.norm(right_side_points_3d - C, dim=-1)
        dist = torch.stack([left_dist, right_dist], dim=-1)
        _, visible_index = torch.min(dist, dim=-1)

        row = torch.arange(visible_index.numel()).type_as(visible_index)
        # may be one of them or may be none of them
        visible_side = side[row, visible_index]

        # img_name = '/data/object/training/image_2/000052.png'
        # draw_line(img_name, visible_side)

        # in abnormal case both of them is invisible
        left_slope = geometry_utils.torch_line_to_orientation(left_side[:, 0],
                                                              left_side[:, 1])
        right_slope = geometry_utils.torch_line_to_orientation(
            right_side[:, 0], right_side[:, 1])
        non_visible_cond = left_slope * right_slope < 0

        visible_slope = geometry_utils.torch_line_to_orientation(
            visible_side[:, 0], visible_side[:, 1])
        # cls_orients
        cls_orients = visible_slope > 0
        cls_orients = cls_orients.float()
        cls_orients[non_visible_cond] = 2.0

        # reg_orients
        boxes_3d_proj = geometry_utils.torch_corners_2d_to_boxes_2d(corners_2d)
        # shape(N, 4)
        boxes_3d_proj_xywh = geometry_utils.torch_xyxy_to_xywh(
            boxes_3d_proj.unsqueeze(0)).squeeze(0)
        direction = torch.abs(visible_side[:, 0] - visible_side[:, 1])
        reg_orients = direction / boxes_3d_proj_xywh[:, 2:]

        return torch.cat([cls_orients.unsqueeze(-1), reg_orients], dim=-1)

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
    def decode_batch(orients, rcnn_proposals, p2):
        """
        Note that rcnn_proposals also refers to boxes_3d_proj
        Args:
            orients: shape(N, )
        """

        assert orients.shape[-1] == 5
        cls_orients = orients[:, :, :3]
        reg_orients = orients[:, :, 3:]
        cls_orients = F.softmax(cls_orients, dim=-1)
        _, cls_orients_argmax = torch.max(cls_orients, keepdim=True, dim=-1)

        rcnn_proposals_xywh = geometry_utils.torch_xyxy_to_xywh(rcnn_proposals)
        reg_orients = reg_orients * rcnn_proposals_xywh[:, :, 2:]

        orients = torch.cat(
            [cls_orients_argmax.type_as(reg_orients), reg_orients], dim=-1)

        side_points = OrientsCoder._generate_side_points(rcnn_proposals,
                                                         orients)

        ry = geometry_utils.torch_pts_2d_to_dir_3d(side_points, p2)

        return ry

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
        abnormal_cond = cls_orient_argmax == 2
        cls_orient_argmax[cls_orient_argmax == 2] = 0
        reg_orient = orient[:, :, 1:3]
        # abnormal_case = torch.cat(
        # [dets_2d[:, :, :2], dets_2d[:, :, :2] + reg_orient], dim=-1)
        abnormal_case = torch.cat(
            [dets_2d[:, :, 2:] - reg_orient, dets_2d[:, :, 2:]], dim=-1)

        side_points = torch.zeros(
            (orient.shape[0], orient.shape[1], 4)).type_as(orient)

        # cls_orient_argmax = np.argmax(cls_orient, axis=-1)

        row_inds = torch.arange(0, cls_orient_argmax.numel()).long()

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

        # add abnormal case
        side_points[abnormal_cond] = abnormal_case[abnormal_cond]
        return side_points
