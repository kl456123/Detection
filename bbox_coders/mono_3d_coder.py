# -*- coding: utf-8 -*-
import torch

from utils.registry import BBOX_CODERS
from core import constants
from utils import geometry_utils
import torch.nn.functional as F

from core.utils import format_checker


@BBOX_CODERS.register(constants.KEY_MONO_3D_NON_2D_PROJ)
class Corner2DNearestCoder(object):
    @staticmethod
    def decode_batch(encoded_corners_2d_all, final_boxes_2d, p2):
        """
        Args:
            encoded_all: shape(N,M, 2+1+4)
        """
        # import ipdb
        # ipdb.set_trace()
        N, M, K = encoded_corners_2d_all.shape
        left_top = final_boxes_2d[:, :, :2]
        final_boxes_2d_xywh = geometry_utils.torch_xyxy_to_xywh(final_boxes_2d)
        wh = final_boxes_2d_xywh[:, :, 2:]

        N, M = encoded_corners_2d_all.shape[:2]
        C_2d = encoded_corners_2d_all[:, :, :2]
        C_2d = C_2d * wh + left_top
        depth_instance = encoded_corners_2d_all[:, :, 2:3]
        location = geometry_utils.torch_points_2d_to_points_3d(
            C_2d, depth_instance, p2)

        # get orients
        bottom_corners = encoded_corners_2d_all[:, :, 3:]
        bottom_corners = bottom_corners * wh + left_top
        bottom_corners = bottom_corners.view(N, M, 4, 2)

        ry_left = geometry_utils.torch_pts_2d_to_dir_3d(
            bottom_corners[:, :, [0, 3]], p2)
        ry_right = geometry_utils.torch_pts_2d_to_dir_3d(
            bottom_corners[:, :, [1, 2]], p2)
        ry = (ry_left + ry_right) / 2

        format_checker.check_tensor_shape(C_2d, [None, None, 2])
        format_checker.check_tensor_shape(depth_instance, [None, None, 1])
        format_checker.check_tensor_shape(bottom_corners, [None, None, 8])

        return torch.stack([location, ry], dim=-1)

    #  @staticmethod
    #  def reorder_boxes_4c_decode(boxes_4c):
        #  """
        #  Args:
            #  boxes_4c: shape(N, M, 4, 2)
        #  """
        #  cond = (boxes_4c[:, :, 0, 1] - boxes_4c[:, :, 1, 1]) < 0
        #  new_boxes_4c = torch.clone(boxes_4c)
        #  new_boxes_4c[cond] = boxes_4c[cond][:, [1, 0, 3, 2]]
        #  return new_boxes_4c

    #  @staticmethod
    #  def reorder_boxes_4c_encode(boxes_4c):
        #  """
        #  Reorder boxes_4c by clockwise
        #  Args:
            #  boxes_4c: shape(N, 4, 2)
        #  """
        #  cond = (boxes_4c[:, 1, 0] - boxes_4c[:, 0, 0]) > 0
        #  new_boxes_4c = torch.clone(boxes_4c)
        #  # import ipdb
        #  # ipdb.set_trace()
        #  new_boxes_4c[cond] = boxes_4c[cond][:, [1, 0, 3, 2]]
        #  return new_boxes_4c

    #  @staticmethod
    #  def encode_with_bbox(boxes_4c, label_boxes_2d):
        #  """
        #  start from right down, ordered by clockwise
        #  Args:
            #  plane_2d: shape(N, 4, 2)
            #  label_boxes_2d: shape(N, 4)
        #  """
        #  # import ipdb
        #  # ipdb.set_trace()
        #  # extend bbox to box_4c
        #  left_top = label_boxes_2d[:, :2]
        #  right_down = label_boxes_2d[:, 2:]
        #  left_down = label_boxes_2d[:, [0, 3]]
        #  right_top = label_boxes_2d[:, [2, 1]]
        #  label_boxes_4c = torch.stack(
            #  [right_down, left_down, left_top, right_top], dim=1)
        #  # label_boxes_4c = torch.stack(
        #  # [left_top, left_top, left_top, left_top], dim=1)

        #  label_boxes_2d_xywh = geometry_utils.torch_xyxy_to_xywh(
            #  label_boxes_2d.unsqueeze(0)).squeeze(0)

        #  # ordered like label_boxes_4c
        #  # import ipdb
        #  # ipdb.set_trace()
        #  boxes_4c = Corner2DNearestCoder.reorder_boxes_4c_encode(boxes_4c)
        #  return (boxes_4c - label_boxes_4c
                #  ) / label_boxes_2d_xywh[:, 2:].unsqueeze(1), boxes_4c

    #  @staticmethod
    #  def decode_with_bbox(encoded_boxes_4c, label_boxes_2d):
        #  """
        #  start from right down, ordered by clockwise
        #  Args:
            #  plane_2d: shape(N, 4, 2)
            #  label_boxes_2d: shape(N, 4)
        #  """
        #  # extend bbox to box_4c
        #  left_top = label_boxes_2d[:, :, :2]
        #  right_down = label_boxes_2d[:, :, 2:]
        #  left_down = label_boxes_2d[:, :, [0, 3]]
        #  right_top = label_boxes_2d[:, :, [2, 1]]

        #  label_boxes_2d_xywh = geometry_utils.torch_xyxy_to_xywh(label_boxes_2d)
        #  label_boxes_4c = torch.stack(
            #  [right_down, left_down, left_top, right_top], dim=2)
        #  # label_boxes_4c = torch.stack(
        #  # [left_top, left_top, left_top, left_top], dim=2)
        #  return encoded_boxes_4c * label_boxes_2d_xywh[:, :, 2:].unsqueeze(
            #  -2) + label_boxes_4c

    @staticmethod
    def encode(label_boxes_3d, label_boxes_2d, p2):
        """
        Args:
            label_boxes_3d: shape(N, K)
        Returns:
            C_2d: shape(N, 2)
            depth: shape(N, )
            side_points_2d: shape(N, 2, 2)
        """
        import ipdb
        ipdb.set_trace()
        num_samples = label_boxes_3d.shape[0]
        location = label_boxes_3d[:, :3]
        C_2d = geometry_utils.torch_points_3d_to_points_2d(location, p2)
        instance_depth = location[:, 2]

        # get side points (two side, yep we predict both of them)
        corners_2d = geometry_utils.torch_boxes_3d_to_corners_2d(
            label_boxes_3d, p2)
        bottom_corners = corners_2d[:, [0, 1, 2, 3]]
        #  left_side = corners_2d[:,[0,3]]
        #  right_side = corners_2d[:,[1,2]]

        encoded_all = torch.cat(
            [C_2d, instance_depth, bottom_corners.view(num_samples, -1)],
            dim=-1)
        return encoded_all

    @staticmethod
    def encode_batch(label_boxes_3d, label_boxes_2d, p2):
        batch_size = label_boxes_3d.shape[0]
        orients_batch = []
        for batch_ind in range(batch_size):
            orients_batch.append(
                Corner2DNearestCoder.encode(
                    label_boxes_3d[batch_ind], label_boxes_2d[batch_ind],
                    p2[batch_ind]))
        return torch.stack(orients_batch, dim=0)
