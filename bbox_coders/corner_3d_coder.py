import torch

from utils.registry import BBOX_CODERS
from core import constants
from utils import geometry_utils

from core.utils import format_checker


@BBOX_CODERS.register(constants.KEY_CORNERS_3D)
class Corner3DCoder(object):
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
    def decode(encoded_corners_3d_all, final_boxes_2d, p2):
        """
        """

        # local to global
        local_corners_3d = encoded_corners_3d_all[:, :24]
        encoded_C_2d = encoded_corners_3d_all[:, 24:26]
        instance_depth_inv = encoded_corners_3d_all[:, 26:]

        # decode them first
        instance_depth = 1 / (instance_depth_inv + 1e-8)
        final_boxes_2d_xywh = geometry_utils.torch_xyxy_to_xywh(
            final_boxes_2d.unsqueeze(0)).squeeze(0)
        C_2d = encoded_C_2d * final_boxes_2d_xywh[:,
                                                  2:] + final_boxes_2d_xywh[:, :
                                                                            2]

        # camera view angle
        alpha = geometry_utils.compute_ray_angle(
            C_2d.unsqueeze(0), p2.unsqueeze(0)).squeeze(0)
        R = geometry_utils.torch_ry_to_rotation_matrix(
            alpha.view(-1)).type_as(encoded_corners_3d_all)
        R_inv = torch.inverse(R)

        # loop here
        C = geometry_utils.torch_points_2d_to_points_3d(C_2d, instance_depth,
                                                        p2)
        local_corners_3d = local_corners_3d.view(-1, 8, 3).permute(0, 2, 1)
        global_corners_3d = torch.matmul(R_inv,
                                         local_corners_3d) + C.unsqueeze(-1)
        return global_corners_3d.permute(0, 2, 1)

    @staticmethod
    def encode(label_boxes_3d, label_boxes_2d, p2):
        """
            projection points of 3d bbox center and its corners_3d in local
            coordinates frame
        """
        # global to local
        global_corners_3d = geometry_utils.torch_boxes_3d_to_corners_3d(
            label_boxes_3d)
        C = label_boxes_3d[:, :3]

        # proj of 3d bbox center
        C_2d = geometry_utils.torch_points_3d_to_points_2d(C, p2)

        alpha = geometry_utils.compute_ray_angle(
            C_2d.unsqueeze(0), p2.unsqueeze(0)).squeeze(0)
        R = geometry_utils.torch_ry_to_rotation_matrix(alpha).type_as(
            global_corners_3d)

        # local coords
        num_boxes = global_corners_3d.shape[0]
        local_corners_3d = torch.matmul(
            R, global_corners_3d.permute(0, 2, 1) - C.unsqueeze(-1)).permute(
                0, 2, 1).contiguous().view(num_boxes, -1)

        # instance depth
        instance_depth = C[:, -1:]

        # finally encode them(local_corners_3d is encoded already)
        # C_2d is encoded by center of 2d bbox
        # this func supports batch format only
        label_boxes_2d_xywh = geometry_utils.torch_xyxy_to_xywh(
            label_boxes_2d.unsqueeze(0)).squeeze(0)
        encoded_C_2d = (
            C_2d - label_boxes_2d_xywh[:, :2]) / label_boxes_2d_xywh[:, 2:]

        # instance_depth is encoded just by inverse it
        instance_depth_inv = 1 / instance_depth

        return torch.cat([local_corners_3d, encoded_C_2d, instance_depth_inv],
                         dim=-1)

    @staticmethod
    def encode_batch(label_boxes_3d, label_boxes_2d, p2):
        batch_size = label_boxes_3d.shape[0]
        orients_batch = []
        for batch_ind in range(batch_size):
            orients_batch.append(
                Corner3DCoder.encode(label_boxes_3d[batch_ind],
                                     label_boxes_2d[batch_ind], p2[batch_ind]))
        return torch.stack(orients_batch, dim=0)
