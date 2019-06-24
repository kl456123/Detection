import torch
import math

from utils.registry import BBOX_CODERS
from core import constants
from utils import geometry_utils

from core.utils import format_checker

enable_local_coord = False


@BBOX_CODERS.register(constants.KEY_CORNERS_3D_GRNET)
class Corner3DCoder(object):
    @staticmethod
    def decode_batch_new(encoded_corners_2d_all, final_boxes_2d, p2):
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
        # encoded_corners_2d_all = encoded_corners_2d_all.view(N, M, 8, 4)
        # encoded_corners_2d = encoded_corners_2d_all[:, :, :, :2].contiguous(
        # ).view(N, M, -1)
        # visibility = encoded_corners_2d_all[:, :, :, 2:].contiguous().view(
        # N, M, -1)

        encoded_corners_2d = encoded_corners_2d_all[:, :, :16]

        format_checker.check_tensor_shape(encoded_corners_2d, [None, None, 16])
        # format_checker.check_tensor_shape(visibility, [None, None, 16])
        format_checker.check_tensor_shape(final_boxes_2d, [None, None, 4])

        batch_size = encoded_corners_2d.shape[0]
        num_boxes = encoded_corners_2d.shape[1]

        final_boxes_2d_xywh = geometry_utils.torch_xyxy_to_xywh(final_boxes_2d)
        # left_top = final_boxes_2d[:, :, :2].unsqueeze(2)
        mid = final_boxes_2d_xywh[:, :, :2].unsqueeze(2)
        wh = final_boxes_2d_xywh[:, :, 2:].unsqueeze(2)
        encoded_corners_2d = encoded_corners_2d.view(batch_size, num_boxes, 8,
                                                     2)
        # visibility = visibility.view(batch_size, num_boxes, 8, 2)
        # visibility = F.softmax(visibility, dim=-1)[:, :, :, 1]
        corners_2d = encoded_corners_2d * wh + mid

        # remove invisibility points
        # import ipdb
        # ipdb.set_trace()
        # corners_2d[visibility > 0.5] = -1
        # .view(batch_size, num_boxes, -1)
        return corners_2d

    @staticmethod
    def decode_batch(encoded_corners_3d_all, final_boxes_2d, p2):
        batch_size = encoded_corners_3d_all.shape[0]
        orients_batch = []
        for batch_ind in range(batch_size):
            orients_batch.append(
                Corner3DCoder.decode(encoded_corners_3d_all[batch_ind],
                                     final_boxes_2d[batch_ind], p2[batch_ind]))
        return torch.stack(orients_batch, dim=0)

    @staticmethod
    def calc_local_corners(dims, ry):
        h = dims[:, 0]
        w = dims[:, 1]
        l = dims[:, 2]
        # zeros = torch.zeros_like(l).type_as(l)
        # rotation_matrix = geometry_utils.torch_ry_to_rotation_matrix(ry)
        zeros = torch.zeros_like(ry[:, 0])
        ones = torch.ones_like(ry[:, 0])
        cos = torch.cos(ry[:, 0])
        sin = torch.sin(ry[:, 0])
        # norm = torch.norm(ry, dim=-1)
        # cos = cos / norm
        # sin = sin / norm
        rotation_matrix = torch.stack(
            [cos, zeros, sin, zeros, ones, zeros, -sin, zeros, cos],
            dim=-1).reshape(-1, 3, 3)

        x_corners = torch.stack(
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            dim=0)
        y_corners = torch.stack(
            [zeros, zeros, zeros, zeros, -h, -h, -h, -h], dim=0)
        z_corners = torch.stack(
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            dim=0)

        # shape(N, 3, 8)
        box_points_coords = torch.stack(
            (x_corners, y_corners, z_corners), dim=0)
        # rotate and translate
        # shape(N, 3, 8)
        corners_3d = torch.bmm(rotation_matrix,
                               box_points_coords.permute(2, 0, 1))

        return corners_3d.permute(0, 2, 1)

    @staticmethod
    def decode_center_depth(dims_preds, final_boxes_2d_xywh, p2):
        f = p2[0, 0]
        h_2d = final_boxes_2d_xywh[:, -1]
        h_3d = dims_preds[:, 0]
        depth_preds = f * h_3d / h_2d
        return depth_preds

    @staticmethod
    def decode(encoded_corners_3d_all, final_boxes_2d, p2):
        """
        """

        # import ipdb
        # ipdb.set_trace()
        # local to global
        # local_corners_3d = encoded_corners_3d_all[:, :24]
        # encoded_C_2d = encoded_corners_3d_all[:, 24:26]
        # instance_depth = encoded_corners_3d_all[:, 26:]
        mean_dims = torch.tensor([1.8, 1.8, 3.7]).type_as(final_boxes_2d)
        dims_preds = torch.exp(encoded_corners_3d_all[:, :3]) * mean_dims
        ry_preds = encoded_corners_3d_all[:, 3:4]

        local_corners_3d = Corner3DCoder.calc_local_corners(
            dims_preds, ry_preds)
        encoded_C_2d = encoded_corners_3d_all[:, 4:6]
        # instance_depth = encoded_corners_3d_all[:, 7:]

        # decode them first
        # instance_depth = 1 / (instance_depth_inv + 1e-8)
        final_boxes_2d_xywh = geometry_utils.torch_xyxy_to_xywh(
            final_boxes_2d.unsqueeze(0)).squeeze(0)
        instance_depth = Corner3DCoder.decode_center_depth(
            dims_preds, final_boxes_2d_xywh, p2)
        C_2d = encoded_C_2d * final_boxes_2d_xywh[:,
                                                  2:] + final_boxes_2d_xywh[:, :
                                                                            2]

        C = geometry_utils.torch_points_2d_to_points_3d(
            C_2d, instance_depth, p2)
        local_corners_3d = local_corners_3d.view(-1, 8, 3).permute(0, 2, 1)
        if enable_local_coord:
            # camera view angle
            alpha = geometry_utils.compute_ray_angle(
                C_2d.unsqueeze(0), p2.unsqueeze(0)).squeeze(0)

            # loop here

            R_inv = geometry_utils.torch_ry_to_rotation_matrix(
                alpha.view(-1)).type_as(encoded_corners_3d_all)
            global_corners_3d = torch.matmul(
                R_inv, local_corners_3d) + C.unsqueeze(-1)
        else:
            # may be slow
            # R_inv = torch.inverse(R)
            global_corners_3d = local_corners_3d + C.unsqueeze(-1)
        return global_corners_3d.permute(0, 2, 1)

    @staticmethod
    def encode(label_boxes_3d, p2):
        """
            projection points of 3d bbox center and its corners_3d in local
            coordinates frame

        Returns:
            depth of center:
            center 3d location:
            local_corners:
        """
        # global to local
        global_corners_3d = geometry_utils.torch_boxes_3d_to_corners_3d(
            label_boxes_3d)
        location = label_boxes_3d[:, :3]

        num_boxes = global_corners_3d.shape[0]

        if enable_local_coord:
            # proj of 3d bbox center
            location_2d = geometry_utils.torch_points_3d_to_points_2d(
                location, p2)

            alpha = geometry_utils.compute_ray_angle(
                location_2d.unsqueeze(0), p2.unsqueeze(0)).squeeze(0)
            R = geometry_utils.torch_ry_to_rotation_matrix(-alpha).type_as(
                global_corners_3d)
            local_corners_3d = torch.matmul(
                R,
                global_corners_3d.permute(0, 2, 1) -
                location.unsqueeze(-1)).permute(0, 2, 1).contiguous().view(
                    num_boxes, -1)
        else:
            # local coords
            local_corners_3d = (global_corners_3d.permute(0, 2, 1) -
                                location.unsqueeze(-1)).permute(
                                    0, 2, 1).contiguous().view(num_boxes, -1)

        # instance depth
        # instance_depth = location[:, -1:]
        dims = label_boxes_3d[:, 3:6]

        return torch.cat([local_corners_3d, location, dims], dim=-1)

    @staticmethod
    def encode_batch(label_boxes_3d, p2):
        batch_size = label_boxes_3d.shape[0]
        orients_batch = []
        for batch_ind in range(batch_size):
            orients_batch.append(
                Corner3DCoder.encode(label_boxes_3d[batch_ind], p2[batch_ind]))
        return torch.stack(orients_batch, dim=0)
