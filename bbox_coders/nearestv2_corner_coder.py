# -*- coding: utf-8 -*-
import torch

from utils.registry import BBOX_CODERS
from core import constants
from utils import geometry_utils
import torch.nn.functional as F

from core.utils import format_checker


@BBOX_CODERS.register(constants.KEY_CORNERS_2D_NEARESTV2)
class NearestV2CornerCoder(object):
    @staticmethod
    def decode_batch(encoded_corners_2d_all, final_boxes_2d):
        """
        Args:
            encoded_corners_2d: shape(N, M, 8 * 4)
            visibility: shape(N, M, 8*2)
            final_bboxes_2d: shape(N, M, 4)
        Returns:
            corners_2d: shape(N, M, 8, 2)
        """
        N, M = encoded_corners_2d_all.shape[:2]
        # encoded_corners_2d = torch.cat([encoded_corners_2d_all[:,:,::4],encoded_corners_2d_all[:,:,1::4]],dim=-1)
        # visibility = torch.cat([encoded_corners_2d_all[:,:,2::4],encoded_corners_2d_all[:,:,3::4]],dim=-1)
        encoded_corners_2d_all = encoded_corners_2d_all.view(N, M, 8, 4)
        encoded_corners_2d = encoded_corners_2d_all[:, :, :, :2].contiguous(
        ).view(N, M, -1)
        visibility = encoded_corners_2d_all[:, :, :, 2:].contiguous().view(
            N, M, -1)

        format_checker.check_tensor_shape(encoded_corners_2d, [None, None, 16])
        format_checker.check_tensor_shape(visibility, [None, None, 16])
        format_checker.check_tensor_shape(final_boxes_2d, [None, None, 4])

        batch_size = encoded_corners_2d.shape[0]
        num_boxes = encoded_corners_2d.shape[1]

        final_boxes_2d_xywh = geometry_utils.torch_xyxy_to_xywh(final_boxes_2d)
        left_top = final_boxes_2d[:, :, :2].unsqueeze(2)
        # mid = final_boxes_2d_xywh[:, :, :2].unsqueeze(2)
        wh = final_boxes_2d_xywh[:, :, 2:].unsqueeze(2)
        encoded_corners_2d = encoded_corners_2d.view(batch_size, num_boxes, 8,
                                                     2)
        visibility = visibility.view(batch_size, num_boxes, 8, 2)
        visibility = F.softmax(visibility, dim=-1)[:, :, :, 1]
        corners_2d = encoded_corners_2d * wh + left_top

        # remove invisibility points
        # import ipdb
        # ipdb.set_trace()
        # corners_2d[visibility > 0.5] = -1
        # .view(batch_size, num_boxes, -1)
        return corners_2d

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
    def reorder_boxes_4c(corners_2d):
        """
        Args:
            corners_2d: shape(N, 8, 2)
        Returns:
            pass
        """
        bottom_corners = corners_2d[:, [0, 1, 2, 3]]
        top_corners = corners_2d[:, [4, 5, 6, 7]]

        def reorder_part_corners(bottom_corners):
            left_side = bottom_corners[:, [0,3]]
            right_side = bottom_corners[:,[1,2]]

            # reorder side by y
            # cond = (left_side[:, 0, 1] - right_side[:, 1, 1])>0
            _, left_order = torch.sort(left_side[:, :, 1], descending=True, dim=-1)
            left_side = tensor_utils.multidim_index(left_side, order)

            _, right_order = torch.sort(right_side[:, :, 1], descending=True, dim=-1)
            right_side = tensor_utils.multidim_index(right_side, order)


        bottom_corners, order = reorder_part_corners(bottom_corners)
        top_corners = tensor_utils.multidim_index(top_corners, order)

        return corners_2d

    @staticmethod
    def encode(label_boxes_3d, label_boxes_2d, p2, image_info):
        """
            return projections of 3d bbox corners in the inner of 2d bbox.
            Note that set the visibility at the same time according to the 2d bbox
            and image boundary.(truncated or occluded)
        """
        # import ipdb
        # ipdb.set_trace()

        # shape(N, 8, 2)
        corners_3d = geometry_utils.torch_boxes_3d_to_corners_3d(
            label_boxes_3d)
        corners_2d = geometry_utils.torch_points_3d_to_points_2d(
            corners_3d.reshape((-1, 3)), p2).reshape(-1, 8, 2)
        # corners_2d = geometry_utils.torch_boxes_3d_to_corners_2d(
        # label_boxes_3d, p2)
        corners_2d = NearestV2CornerCoder.reorder_boxes_4c(corners_2d)

        image_shape = torch.tensor([0, 0, image_info[1], image_info[0]])
        image_shape = image_shape.type_as(corners_2d).view(1, 4)
        image_filter = geometry_utils.torch_window_filter(
            corners_2d, image_shape, deltas=200)

        boxes_2d_filter = geometry_utils.torch_window_filter(corners_2d,
                                                             label_boxes_2d)

        # disable it at preseant
        self_occluded_filter = Corner2DCoder.get_occluded_filter(corners_3d)
        # self_occluded_filter = torch.ones_like(image_filter)
        # self_occluded_filter = 0.1 * self_occluded_filter.float()

        # points outside of image must be filter out
        visibility = image_filter.float() * self_occluded_filter
        # visibility = visibility & boxes_2d_filter & self_occluded_filter

        # remove invisibility points
        #  corners_2d[~visibility] = -1

        # normalize using label bbox 2d
        label_boxes_2d_xywh = geometry_utils.torch_xyxy_to_xywh(
            label_boxes_2d.unsqueeze(0)).squeeze(0)
        wh = label_boxes_2d_xywh[:, 2:].unsqueeze(1)
        left_top = label_boxes_2d[:, :2].unsqueeze(1)
        # mid = label_boxes_2d_xywh[:, :2].unsqueeze(1)
        encoded_corners_2d = (corners_2d - left_top) / wh

        encoded_corners_2d = torch.cat(
            [encoded_corners_2d, visibility.unsqueeze(-1).float()], dim=-1)
        return encoded_corners_2d.contiguous().view(
            encoded_corners_2d.shape[0], -1)

    @staticmethod
    def encode_batch(label_boxes_3d, label_boxes_2d, p2, image_info):
        batch_size = label_boxes_3d.shape[0]
        orients_batch = []
        for batch_ind in range(batch_size):
            orients_batch.append(
                Corner2DCoder.encode(label_boxes_3d[batch_ind], label_boxes_2d[
                    batch_ind], p2[batch_ind], image_info[batch_ind]))
        return torch.stack(orients_batch, dim=0)
