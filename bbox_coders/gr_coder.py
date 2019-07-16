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
    def decode_batch(encoded_corners_2d_all, final_boxes_2d, p2):
        """
        Args:
            encoded_corners_2d: shape(N, M, 8 * 4)
            visibility: shape(N, M, 8*2)
            final_bboxes_2d: shape(N, M, 4)
        Returns:
            corners_2d: shape(N, M, 8, 2)
        """
        N, M = encoded_corners_2d_all.shape[:2]

        local_corners_3d = 3*torch.tanh(encoded_corners_2d_all[:, :, :24])
        depth = encoded_corners_2d_all[:, :, 24:25]
        center_cylinder_2d = encoded_corners_2d_all[:, :, 25:27]

        # format_checker.check_tensor_shape(encoded_corners_2d, [None, None, 16])
        # format_checker.check_tensor_shape(visibility, [None, None, 16])
        format_checker.check_tensor_shape(final_boxes_2d, [None, None, 4])

        final_boxes_2d_xywh = geometry_utils.torch_xyxy_to_xywh(final_boxes_2d)
        # left_top = final_boxes_2d[:, :, :2].unsqueeze(2)
        mid = final_boxes_2d_xywh[:, :, :2]
        wh = final_boxes_2d_xywh[:, :, 2:]
        center_cylinder_2d = center_cylinder_2d * wh + mid

        #  import ipdb
        #  ipdb.set_trace()
        locations = []
        for batch_ind in range(N):
            locations.append(
                geometry_utils.torch_cylinder_points_2d_to_points_3d_v2(
                    center_cylinder_2d[batch_ind],
                    depth[batch_ind],
                    p2[batch_ind],
                    radus=864))
        locations = torch.stack(locations, dim=0)

        # camera view angle
        ray_angle = -torch.atan2(locations[:, :, 2], locations[:, :, 0])
        #  alpha = geometry_utils.compute_ray_angle(
        #  C_2d.unsqueeze(0), p2.unsqueeze(0)).squeeze(0)

        # loop here

        #  import ipdb
        #  ipdb.set_trace()
        R_inv = geometry_utils.torch_ry_to_rotation_matrix(
            ray_angle.view(-1)).type_as(local_corners_3d)
        local_corners_3d = local_corners_3d.view(-1, 8, 3).permute(0, 2, 1)
        global_corners_3d = torch.bmm(
            R_inv, local_corners_3d) + locations.view(-1, 3, 1)

        return global_corners_3d.permute(0, 2, 1).view(N, M, 8, 3)

    @staticmethod
    def decode_batch_old(encoded_corners_3d_all, final_boxes_2d, p2):
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
    def decode_ry(encoded_ry_preds, proposals_xywh, p2):
        slope, encoded_points = torch.split(encoded_ry_preds, [1, 2], dim=-1)
        slope = slope * proposals_xywh[:, :, 3:4] / (
            proposals_xywh[:, :, 2:3] + 1e-7)
        points1 = encoded_points * proposals_xywh[:, :,
                                                  2:] + proposals_xywh[:, :, :2]
        points2_x = points1[:, :, :1] - 1
        points2_y = points1[:, :, 1:] - slope
        points2 = torch.cat([points2_x, points2_y], dim=-1)
        lines = torch.cat([points1, points2], dim=-1)
        ry = geometry_utils.torch_pts_2d_to_dir_3d(lines, p2)
        return ry

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
        encoded_ry_preds = encoded_corners_3d_all[:, 3:6]
        final_boxes_2d_xywh = geometry_utils.torch_xyxy_to_xywh(
            final_boxes_2d.unsqueeze(0)).squeeze(0)
        ry_preds = Corner3DCoder.decode_ry(
            encoded_ry_preds.unsqueeze(0), final_boxes_2d_xywh.unsqueeze(0),
            p2.unsqueeze(0)).squeeze(0)
        ry_preds = ry_preds.unsqueeze(-1)

        local_corners_3d = Corner3DCoder.calc_local_corners(
            dims_preds, ry_preds)
        encoded_C_2d = encoded_corners_3d_all[:, 6:8]
        # instance_depth = encoded_corners_3d_all[:, 8:]

        # decode them first
        # instance_depth = 1 / (instance_depth_inv + 1e-8)

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
            depth of center: here refers to abs distance between camera center and object center
            local_corners:
        """
        # global to local
        dims = label_boxes_3d[:, 3:6]
        location = label_boxes_3d[:, :3]
        ry = label_boxes_3d[:, -1:]
        ray_angle = -torch.atan2(location[:, 2], location[:, 0])
        local_angle = ry - ray_angle[:, None]
        local_corners_3d = geometry_utils.torch_calc_local_corners(
            dims, local_angle)
        depth = torch.norm(location[..., [0, 2]], dim=-1, keepdim=True)
        # depth = location[:, -1:]

        #  import ipdb
        #  ipdb.set_trace()
        location[:, 1] = location[:, 1] - 0.5 * dims[:, 0]
        center_cylinder_2d = geometry_utils.torch_points_3d_to_cylinder_points_2d(
            location, p2, radus=864)
        local_corners_3d[..., 1] = local_corners_3d[..., 1] + 0.5 * dims[:, None, 0]
        # center_cylinder_2d = geometry_utils.torch_points_3d_to_points_2d(location, p2)
        #  corners_2d = geometry_utils.torch_boxes_3d_to_corners_2d(label_boxes_3d, p2)
        #  boxes_2d = geometry_utils.torch_corners_2d_to_boxes_2d(corners_2d)
        #  center_cylinder_2d = boxes_2d[:, :2]

        return torch.cat(
            [
                local_corners_3d.contiguous().view(-1, 24), depth,
                center_cylinder_2d
            ],
            dim=-1)

    @staticmethod
    def encode_batch(label_boxes_3d, p2):
        batch_size = label_boxes_3d.shape[0]
        orients_batch = []
        for batch_ind in range(batch_size):
            orients_batch.append(
                Corner3DCoder.encode(label_boxes_3d[batch_ind], p2[batch_ind]))
        return torch.stack(orients_batch, dim=0)
