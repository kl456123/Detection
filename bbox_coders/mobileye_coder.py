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
    def decode_batch(encoded_corners_3d_all, proposals):
        """
        Args:
            encoded_corners_3d_all: shape(N, M, 12)
        """
        # import ipdb
        # ipdb.set_trace()
        N, M = encoded_corners_3d_all.shape[:2]
        encoded_corners_3d_all = encoded_corners_3d_all.view(N, M, 8, 2)

        proposals_xywh = geometry_utils.torch_xyxy_to_xywh(proposals)
        corners_2d = encoded_corners_3d_all * proposals_xywh[:, :, 2:].unsqueeze(
            -2) + proposals_xywh[:, :, :2].unsqueeze(-2)
        # corners_2d = torch.cat(
        # [corners_2d[:, :, :2], corners_2d[:, :, 2:4], torch.zeros_like(corners_2d[:, :,4:])], dim=2)
        bottom_corners = corners_2d[:, :, [0, 2, 4, 6]]
        top_corners = corners_2d[:, :, [1, 3, 5, 7]]
        corners_2d = torch.cat([bottom_corners, top_corners], dim=2)
        return corners_2d

    @staticmethod
    def reorder_lines(lines):
        lines = lines.view(-1, 2, 2, 2)
        _, order = torch.sort(lines[:, :, 0, 0], dim=-1, descending=False)
        order_lines = tensor_utils.multidim_index(lines.contiguous().view(
            -1, 2, 4), order)
        return order_lines

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
        # import ipdb
        # ipdb.set_trace()
        N = label_boxes_3d.shape[0]
        corners_2d = geometry_utils.torch_boxes_3d_to_corners_2d(
            label_boxes_3d, p2)
        front_face_lines = corners_2d[:, [0, 4, 1, 5]]
        rear_face_lines = corners_2d[:, [3, 7, 2, 6]]
        front_face_lines = Corner3DCoder.reorder_lines(front_face_lines)
        rear_face_lines = Corner3DCoder.reorder_lines(rear_face_lines)

        reorder_corners_2d = torch.cat(
            [front_face_lines, rear_face_lines], dim=1)

        proposals_xywh = geometry_utils.torch_xyxy_to_xywh(proposals[None])[0]
        encoded_corners_2d = (
            reorder_corners_2d.view(-1, 8, 2) - proposals_xywh[:, :2][:, None]
        ) / proposals_xywh[:, 2:][:, None]

        image_shape = torch.tensor([0, 0, image_info[1], image_info[0]])
        image_shape = image_shape.type_as(corners_2d).view(1, 4)
        image_filter = geometry_utils.torch_window_filter(
            reorder_corners_2d.view(-1, 8, 2), image_shape,
            deltas=200).float()

        total = torch.cat(
            [encoded_corners_2d.view(N, -1),
             image_filter.view(N, -1)], dim=-1)

        return total

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
