import torch
import math

from utils.registry import BBOX_CODERS
from core import constants
from utils import geometry_utils

from core.utils import format_checker


@BBOX_CODERS.register(constants.KEY_MOBILEYE)
class Corner3DCoder(object):
    @staticmethod
    def decode_batch(encoded_corners_3d_all, proposals):
        """
        Args:
            encoded_corners_3d_all: shape(N, M, 12)
        """

        N, M = encoded_corners_3d_all.shape[:2]
        encoded_bottom_corners = encoded_corners_3d_all[:, :, :8]
        encoded_heights = encoded_corners_3d_all[:, :, 8:12]

        proposals_xywh = geometry_utils.torch_xyxy_to_xywh(proposals)
        bottom_corners = encoded_bottom_corners.view(
            N, M, 4, 2) * proposals_xywh[:, :, 2:].unsqueeze(
                -2) + proposals_xywh[:, :, :2].unsqueeze(-2)

        top_corners_y = bottom_corners[...,
                                       1] - encoded_heights * proposals_xywh[:, :,
                                                                             -1].unsqueeze(
                                                                                 -1
                                                                             )
        top_corners_x = bottom_corners[..., 0]
        top_corners = torch.stack([top_corners_x, top_corners_y], dim=-1)

        corners_2d = torch.cat([bottom_corners, top_corners], dim=-2)
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
    def encode(label_boxes_3d, proposals, p2, image_info):
        """
            projection points of 3d bbox center and its corners_3d in local
            coordinates frame

        Returns:
            depth of center:
            center 3d location:
            local_corners:
        """
        # import ipdb
        # ipdb.set_trace()
        num_instances = label_boxes_3d.shape[0]
        # global to local
        corners_2d = geometry_utils.torch_boxes_3d_to_corners_2d(
            label_boxes_3d, p2)
        bottom_corners = corners_2d[:, [0, 1, 2, 3]]
        # side_line = corners_2d[:, [0, 4, 1, 5, 2, 6, 3, 7]]
        # side_line = side_line.view(-1, 4, 2, 2)
        # side_mid = side_line.mean(dim=-2) # shape(N, 4, 2)
        # left_top = proposals[:, :2]
        proposals_xywh = geometry_utils.torch_xyxy_to_xywh(
            proposals.unsqueeze(0)).squeeze(0)
        wh = proposals_xywh[:, 2:].unsqueeze(1)
        xy = proposals_xywh[:, :2].unsqueeze(1)
        encoded_bottom_corners = (bottom_corners - xy) / wh

        # four height in image
        top_corners = corners_2d[:, [4, 5, 6, 7]]
        encoded_heights = (
            bottom_corners[..., 1] - top_corners[..., 1]) / wh[..., 1]

        # import ipdb
        # ipdb.set_trace()
        # visibility
        image_shape = torch.tensor([0, 0, image_info[1], image_info[0]])
        image_shape = image_shape.type_as(corners_2d).view(1, 4)
        image_filter = geometry_utils.torch_window_filter(
            corners_2d[:, :4], image_shape, deltas=200)

        corners_3d = geometry_utils.torch_boxes_3d_to_corners_3d(
            label_boxes_3d)
        self_occluded_filter = Corner3DCoder.get_occluded_filter(corners_3d[:, :4])
        visibility = image_filter.float() * self_occluded_filter

        return torch.cat(
            [
                encoded_bottom_corners.contiguous().view(num_instances, -1),
                encoded_heights.contiguous().view(num_instances, -1), visibility
            ],
            dim=-1)

    @staticmethod
    def encode_batch(label_boxes_3d, proposals, p2, image_info):
        batch_size = label_boxes_3d.shape[0]
        orients_batch = []
        for batch_ind in range(batch_size):
            orients_batch.append(
                Corner3DCoder.encode(label_boxes_3d[batch_ind],
                                     proposals[batch_ind], p2[batch_ind],
                                     image_info[batch_ind]))
        return torch.stack(orients_batch, dim=0)
