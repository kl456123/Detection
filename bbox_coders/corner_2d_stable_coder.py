# -*- coding: utf-8 -*-
import torch

from utils.registry import BBOX_CODERS
from core import constants
from utils import geometry_utils
import torch.nn.functional as F

from core.utils import format_checker
from utils import math_utils
from core.utils import tensor_utils


@BBOX_CODERS.register(constants.KEY_CORNERS_2D_STABLE)
class Corner2DStableCoder(object):
    @staticmethod
    def decode_batch(encoded_corners_2d_all, final_boxes_2d):
        """
        Args:
            encoded_corners_2d: shape(N, M, 8 * (4*2+4))
            final_bboxes_2d: shape(N, M, 4)
        Returns:
            corners_2d: shape(N, M, 8, 2)
        """
        # import ipdb
        # ipdb.set_trace()
        N, M = encoded_corners_2d_all.shape[:2]
        # format_checker.check_tensor_shape(encoded_corners_2d_all,
        # [None, None, None])
        encoded_corners_2d_all = encoded_corners_2d_all.view(N, M, -1)
        encoded_corners_2d = encoded_corners_2d_all[
            ..., :64].contiguous().view(N, M, 8, 4, 2)
        corners_2d_scores = encoded_corners_2d_all[..., 64:].contiguous().view(
            N, M, 8, 4)

        # corners_2d_scores = F.softmax(corners_2d_scores, dim=-1)
        argmax = corners_2d_scores.max(dim=-1)[-1]

        # format_checker.check_tensor_shape(visibility, [None, None, 16])
        format_checker.check_tensor_shape(final_boxes_2d, [None, None, 4])

        batch_size = encoded_corners_2d.shape[0]
        num_boxes = encoded_corners_2d.shape[1]

        final_boxes_2d_xywh = geometry_utils.torch_xyxy_to_xywh(final_boxes_2d)
        # left_top = final_boxes_2d[:, :, :2].unsqueeze(2)
        wh = final_boxes_2d_xywh[:, :, 2:].unsqueeze(2).unsqueeze(2)
        corners_4c = geometry_utils.torch_xyxy_to_corner_4c(final_boxes_2d)

        encoded_corners_2d = encoded_corners_2d.view(batch_size, num_boxes, 8,
                                                     4, 2)
        corners_2d = encoded_corners_2d * wh + corners_4c.unsqueeze(2)

        # sum all
        # corners_2d = corners_2d.mean(dim=3)
        row = torch.arange(argmax.numel()).type_as(argmax)
        corners_2d = corners_2d.view(-1, 4, 2)
        corners_2d = corners_2d[row, argmax.view(-1)]
        # corners_2d = corners_2d[..., 3, :]

        return corners_2d.view(N, M, 8, 2)

    @staticmethod
    def get_occluded_filter(corners_3d):
        # import ipdb
        # ipdb.set_trace()
        _, argmax = torch.max(torch.norm(corners_3d, dim=-1), dim=-1)
        device = corners_3d.device

        occluded_filter = torch.ones(corners_3d.shape[0], 8).to(device).float()
        row = torch.arange(0, corners_3d.shape[0]).type_as(argmax)
        occluded_filter[row, argmax] = 0.1
        return occluded_filter

    @staticmethod
    def encode(label_boxes_3d, proposals, p2, image_info):
        """
        return projections of 3d bbox corners in the inner of 2d bbox.
            Note that set the visibility at the same time according to the 2d bbox
            and image boundary.(truncated or occluded)
        """
        label_boxes_2d = proposals
        # shape(N, 8, 2)
        corners_3d = geometry_utils.torch_boxes_3d_to_corners_3d(
            label_boxes_3d)
        corners_2d = geometry_utils.torch_points_3d_to_points_2d(
            corners_3d.reshape((-1, 3)), p2).reshape(-1, 8, 2)

        image_shape = torch.tensor([0, 0, image_info[1], image_info[0]])
        image_shape = image_shape.type_as(corners_2d).view(1, 4)
        image_filter = geometry_utils.torch_window_filter(
            corners_2d, image_shape, deltas=200)

        # points outside of image must be filter out
        visibility = image_filter.float()

        # normalize using label bbox 2d
        label_boxes_2d_xywh = geometry_utils.torch_xyxy_to_xywh(
            label_boxes_2d.unsqueeze(0)).squeeze(0)
        # shape(N, 4, 2)
        label_corners_4c = geometry_utils.torch_xyxy_to_corner_4c(
            label_boxes_2d.unsqueeze(0)).squeeze(0)
        wh = label_boxes_2d_xywh[:, 2:].unsqueeze(1).unsqueeze(1)
        # left_top = label_boxes_2d[:, :2].unsqueeze(1)
        # mid = label_boxes_2d_xywh[:, :2].unsqueeze(1)
        corners_2d = corners_2d.unsqueeze(2)
        label_corners_4c = label_corners_4c.unsqueeze(1)
        encoded_corners_2d = (corners_2d - label_corners_4c) / wh
        # mean_size = torch.sqrt(wh[..., 0] * wh[..., 1])
        # weights = math_utils.gaussian2d(
        # corners_2d, label_corners_4c, sigma=mean_size)

        # import ipdb
        # ipdb.set_trace()
        dist = torch.norm(encoded_corners_2d, dim=-1)  # (N,8,4)
        dist_min, dist_argmin = dist.min(dim=-1)  # (N,8)
        corners_2d_scores = torch.zeros_like(dist)
        corners_2d_scores = corners_2d_scores.view(-1, 4)
        # offset = torch.arange(dist_argmin.numel()) * 4
        # col_index = dist_argmin.view(-1) + offset.type_as(dist_argmin)
        col_index = dist_argmin.view(-1)
        row_index = torch.arange(col_index.numel()).type_as(col_index)
        corners_2d_scores[row_index, col_index] = 1
        corners_2d_scores = corners_2d_scores.view(-1, 8, 4)
        # tensor_utils.multidim_index(corners_2d_scores, dist_argmin)
        visibility = visibility.unsqueeze(-1) * corners_2d_scores

        # encoded_corners_2d = torch.cat(
        # [
        # encoded_corners_2d,
        # visibility.unsqueeze(-1)
        # # corners_2d_scores.unsqueeze(-1)
        # ],
        # dim=-1)
        # encoded_corners_2d = torch.cat(
        # [
        # encoded_corners_2d.view(encoded_corners_2d.shape[0], 8, -1),
        # dist_argmin.unsqueeze(-1).float()
        # ],
        # dim=-1)
        # encoded_corners_2d = encoded_corners_2d.contiguous().view(
        # encoded_corners_2d.shape[0], -1)
        # import ipdb
        # ipdb.set_trace()
        N = encoded_corners_2d.shape[0]
        return torch.cat(
            [
                encoded_corners_2d.contiguous().view(N, -1),
                visibility.view(N, -1),
                dist_argmin.float().view(N, -1)
            ],
            dim=-1)

    @staticmethod
    def encode_batch(label_boxes_3d, label_boxes_2d, p2, image_info):
        batch_size = label_boxes_3d.shape[0]
        orients_batch = []
        for batch_ind in range(batch_size):
            orients_batch.append(
                Corner2DStableCoder.encode(
                    label_boxes_3d[batch_ind], label_boxes_2d[batch_ind],
                    p2[batch_ind], image_info[batch_ind]))
        return torch.stack(orients_batch, dim=0)
