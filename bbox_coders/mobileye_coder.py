import torch
import math

from utils.registry import BBOX_CODERS
from core import constants
from utils import geometry_utils

from core.utils import format_checker
from core.utils import tensor_utils
import torch.nn.functional as F


@BBOX_CODERS.register(constants.KEY_MOBILEYE)
class Corner3DCoder(object):
    @staticmethod
    def decode_batch(encoded_corners_3d_all, proposals, keypoint_map):
        """
        Args:
            encoded_corners_3d_all: shape(N, M, 12)
        """
        # import ipdb
        # ipdb.set_trace()

        N, M = encoded_corners_3d_all.shape[:2]
        encoded_bottom_corners = encoded_corners_3d_all[:, :, :6]
        encoded_heights = encoded_corners_3d_all[:, :, 6:9]
        keypoint_map = F.softmax(keypoint_map, dim=-1)
        _, keypoint_index = keypoint_map.max(dim=-1)
        resolution = 14
        ratio = keypoint_index.float() / resolution
        proposals_xywh = geometry_utils.torch_xyxy_to_xywh(proposals)
        mid_x = proposals[:, :, 0] + proposals_xywh[:, :, 2] * ratio.view(N,M)

        # visibility = encoded_corners_3d_all[:, :, 9:]
        # invisible_cond = F.softmax(visibility, dim=-1)[..., 1] < 0.5
        # encoded_bottom_corners = encoded_bottom_corners.view(-1, 6)
        # import ipdb
        # ipdb.set_trace()
        # bugs of pytorch !!
        # encoded_bottom_corners[invisible_cond.view(
        # -1)][:, 2:4] = 1
        # tmp = encoded_bottom_corners[invisible_cond.view(-1)]
        # tmp = torch.cat([tmp[:, :2], tmp[:, :2], tmp[:, 4:6]], dim=-1)
        # encoded_bottom_corners[invisible_cond.view(-1)] = tmp
        # encoded_heights[invisible_cond.view(-1), 1] = encoded_heights[invisible_cond.view(-1), 0]
        # encoded_heights[..., 1][invisible_cond.view(-1)] = encoded_heights[..., 0][invisible_cond.view(-1)]

        # reshape back
        # encoded_bottom_corners = encoded_bottom_corners.view(N, M, 3, 2)

        # encoded_heights = encoded_heights.view(-1, 3)
        # tmp = encoded_heights[invisible_cond.view(-1)]
        # tmp = torch.stack([tmp[:, 0], tmp[:, 0], tmp[:, 2]], dim=-1)
        # encoded_heights[invisible_cond.view(-1)] = tmp
        # encoded_heights = encoded_heights.view(N, M, 3)

        bottom_corners = encoded_bottom_corners.view(
            N, M, 3, 2) * proposals_xywh[:, :, 2:].unsqueeze(
                -2) + proposals_xywh[:, :, :2].unsqueeze(-2)


        # use cls results instead
        bottom_corners[:, :, 1, 0] = mid_x
        # import ipdb
        # ipdb.set_trace()

        top_corners_y = bottom_corners[...,
                                       1] - encoded_heights * proposals_xywh[:, :,
                                                                             -1].unsqueeze(
                                                                                 -1
                                                                             )
        top_corners_x = bottom_corners[..., 0]
        top_corners = torch.stack([top_corners_x, top_corners_y], dim=-1)

        # append others
        # corners_2d = torch.cat(
        # [corners_2d, corners_2d[:, :, :1], corners_2d[:, :, 3:4]], dim=-2)

        corners_2d = torch.cat(
            [
                bottom_corners, bottom_corners[:, :, :1], top_corners,
                top_corners[:, :, :1]
            ],
            dim=-2)
        return corners_2d

    @staticmethod
    def decode(encoded_corners_3d_all, proposals, p2):
        """
        """
        pass

    @staticmethod
    def get_occluded_filter(corners_3d):
        # import ipdb
        # ipdb.set_trace()
        _, argmax = torch.max(torch.norm(corners_3d, dim=-1), dim=-1)
        device = corners_3d.device

        occluded_filter = torch.ones(corners_3d.shape[0], 4).to(device).float()
        row = torch.arange(0, corners_3d.shape[0]).type_as(argmax)
        occluded_filter[row, argmax] = 0
        return occluded_filter

    @staticmethod
    def find_visible_side(bottom_corners_3d):
        """
        Args:
            corners_3d: shape(N, 4, 3)
        """
        # start_corners_3d = bottom_corners_3d
        dist = torch.norm(bottom_corners_3d, dim=-1)
        _, order = torch.sort(dist, dim=-1, descending=False)
        # visible_corners_3d = tensor_utils.multidim_index(
        # bottom_corners_3d, order[:, :3])
        return order[:, :3]

    @staticmethod
    def encode(label_boxes_3d, proposals, p2, image_info, label_boxes_2d):
        """
            projection points of 3d bbox center and its corners_3d in local
            coordinates frame

        Returns:
            depth of center:
            center 3d location:
            local_corners:
        """
        num_instances = label_boxes_3d.shape[0]
        # global to local
        corners_2d = geometry_utils.torch_boxes_3d_to_corners_2d(
            label_boxes_3d, p2)

        proposals_xywh = geometry_utils.torch_xyxy_to_xywh(
            proposals.unsqueeze(0)).squeeze(0)
        wh = proposals_xywh[:, 2:].unsqueeze(1)
        xy = proposals_xywh[:, :2].unsqueeze(1)

        corners_3d = geometry_utils.torch_boxes_3d_to_corners_3d(
            label_boxes_3d)
        bottom_corners_3d = corners_3d[:, [0, 1, 2, 3]]
        visible_index = Corner3DCoder.find_visible_side(bottom_corners_3d)
        visible_corners_3d = tensor_utils.multidim_index(
            bottom_corners_3d, visible_index)
        visible_side_line_2d = geometry_utils.torch_points_3d_to_points_2d(
            visible_corners_3d.contiguous().view(-1, 3), p2).view(
                num_instances, -1, 2)
        visible_cond = (
            visible_side_line_2d[:, 1, 0] - visible_side_line_2d[:, 0, 0]
        ) * (visible_side_line_2d[:, 2, 0] - visible_side_line_2d[:, 0, 0]) < 0

        # visible_index[invisible_cond, -1] = visible_index[invisible_cond, -2]
        _, order = torch.sort(
            visible_side_line_2d[..., 0], dim=-1, descending=False)
        visible_index = tensor_utils.multidim_index(
            visible_index.unsqueeze(-1), order).squeeze(-1)

        # import ipdb
        # ipdb.set_trace()
        bottom_corners = corners_2d[:, [0, 1, 2, 3]]
        top_corners = corners_2d[:, [4, 5, 6, 7]]
        bottom_corners = tensor_utils.multidim_index(bottom_corners,
                                                     visible_index)
        top_corners = tensor_utils.multidim_index(top_corners, visible_index)

        # box truncated
        # import ipdb
        # ipdb.set_trace()
        # bottom
        # left
        bottom_corners[:, 0, 0] = torch.min(bottom_corners[:, 0, 0],
                                            label_boxes_2d[:, 2])
        bottom_corners[:, 0, 0] = torch.max(bottom_corners[:, 0, 0],
                                            label_boxes_2d[:, 0])

        # right
        bottom_corners[:, 2, 0] = torch.min(bottom_corners[:, 2, 0],
                                            label_boxes_2d[:, 2])
        bottom_corners[:, 2, 0] = torch.max(bottom_corners[:, 2, 0],
                                            label_boxes_2d[:, 0])

        # top
        top_corners[:, 0, 0] = torch.min(top_corners[:, 0, 0],
                                         label_boxes_2d[:, 2])
        top_corners[:, 0, 0] = torch.max(top_corners[:, 0, 0],
                                         label_boxes_2d[:, 0])

        top_corners[:, 2, 0] = torch.min(top_corners[:, 2, 0],
                                         label_boxes_2d[:, 2])
        top_corners[:, 2, 0] = torch.max(top_corners[:, 2, 0],
                                         label_boxes_2d[:, 0])

        in_box_cond = (bottom_corners[:, 1, 0] < label_boxes_2d[:, 2]) & (
            bottom_corners[:, 1, 0] > label_boxes_2d[:, 0])

        # bottom_corners[:, [0, 2], 0] = bottom_corners[:, [0, 2], 0]
        # top_corners[:, :, 0] = top_corners[:, :, 0].clamp(
        # min=0, max=image_info[1])

        visibility = visible_cond.float() * in_box_cond.float()
        # import ipdb
        # ipdb.set_trace()
        index = torch.nonzero(visibility <= 0).view(-1)
        tmp = bottom_corners[index]
        tmp = torch.stack([tmp[:, 0], tmp[:, 0], tmp[:, 2]], dim=1)
        bottom_corners[index] = tmp

        tmp = top_corners[index]
        tmp = torch.stack([tmp[:, 0], tmp[:, 0], tmp[:, 2]], dim=1)
        top_corners[index] = tmp

        # encode
        encoded_bottom_corners = (bottom_corners - xy) / wh
        encoded_heights = (
            bottom_corners[..., 1] - top_corners[..., 1]) / wh[..., 1]

        # import ipdb
        # ipdb.set_trace()
        mid_x = bottom_corners[:, 1, 0]
        ratio = (mid_x - proposals[:, 0]) / wh[:, 0, 0]
        ratio = ratio.clamp(min=0, max=1)

        # import ipdb
        # ipdb.set_trace()
        # encoded_bottom_corners = tensor_utils.multidim_index(
        # encoded_bottom_corners, visible_index)
        # encoded_heights = tensor_utils.multidim_index(
        # encoded_heights.unsqueeze(-1), visible_index)
        # tensor_utils.
        # visibility = tensor_utils.multidim_index(
        # visibility.unsqueeze(-1), visible_index).squeeze(-1)

        return torch.cat(
            [
                encoded_bottom_corners.contiguous().view(num_instances, -1),
                encoded_heights.contiguous().view(num_instances, -1),
                ratio.view(num_instances, -1)
            ],
            dim=-1)

    @staticmethod
    def encode_batch(label_boxes_3d, proposals, p2, image_info,
                     label_boxes_2d):
        batch_size = label_boxes_3d.shape[0]
        orients_batch = []
        for batch_ind in range(batch_size):
            orients_batch.append(
                Corner3DCoder.encode(label_boxes_3d[batch_ind],
                                     proposals[batch_ind], p2[batch_ind],
                                     image_info[batch_ind],
                                     label_boxes_2d[batch_ind]))
        return torch.stack(orients_batch, dim=0)
