# -*- coding: utf-8 -*-
import numpy as np
import torch


class GridAnchor3dGenerator(object):
    def __init__(self, anchor_generator_config):
        self.area_extents = anchor_generator_config['area_extents']
        self.anchor_size = anchor_generator_config['anchor_size']
        self.anchor_stride = anchor_generator_config['anchor_stride']

    def generate_np(self, ground_plane):

        ground_plane = ground_plane.cpu().numpy()
        assert ground_plane.shape[0] == 1, 'Only one plane is supported now'
        ground_plane = ground_plane[0]

        area_extents = self.area_extents
        anchor_3d_sizes = self.anchor_size
        anchor_stride = self.anchor_stride

        # Convert sizes to ndarray
        anchor_3d_sizes = np.asarray(anchor_3d_sizes)

        anchor_stride_x = anchor_stride[0]
        anchor_stride_z = anchor_stride[1]
        anchor_rotations = np.asarray([0, np.pi / 2.0])

        x_start = area_extents[0][0] + anchor_stride[0] / 2.0
        x_end = area_extents[0][1]
        x_centers = np.array(
            np.arange(
                x_start, x_end, step=anchor_stride_x), dtype=np.float32)

        z_start = area_extents[2][1] - anchor_stride[1] / 2.0
        z_end = area_extents[2][0]
        z_centers = np.array(
            np.arange(
                z_start, z_end, step=-anchor_stride_z), dtype=np.float32)

        # Use ranges for substitution
        size_indices = np.arange(0, len(anchor_3d_sizes))

        rotation_indices = np.arange(0, len(anchor_rotations))

        # Generate matrix for substitution
        # e.g. for two sizes and two rotations
        # [[x0, z0, 0, 0], [x0, z0, 0, 1], [x0, z0, 1, 0], [x0, z0, 1, 1],
        #  [x1, z0, 0, 0], [x1, z0, 0, 1], [x1, z0, 1, 0], [x1, z0, 1, 1], ...]
        before_sub = np.stack(
            np.meshgrid(x_centers, z_centers, size_indices, rotation_indices),
            axis=4).reshape(-1, 4)

        # Place anchors on the ground plane
        a, b, c, d = ground_plane
        all_x = before_sub[:, 0]
        all_z = before_sub[:, 1]
        all_y = -(a * all_x + c * all_z + d) / b

        # Create empty matrix to return
        num_anchors = len(before_sub)
        all_anchor_boxes_3d = np.zeros((num_anchors, 7))

        # Fill in x, y, z
        all_anchor_boxes_3d[:, 0:3] = np.stack((all_x, all_y, all_z), axis=1)

        # Fill in shapes
        sizes = anchor_3d_sizes[np.asarray(before_sub[:, 2], np.int32)]
        all_anchor_boxes_3d[:, 3:6] = sizes

        # Fill in rotations
        rotations = anchor_rotations[np.asarray(before_sub[:, 3], np.int32)]
        all_anchor_boxes_3d[:, 6] = rotations

        return all_anchor_boxes_3d

    def generate_pytorch(self, ground_plane):
        # ground_plane = ground_plane.cpu().numpy()
        all_anchor_boxes_3d = self.generate_np(ground_plane)
        return torch.tensor(all_anchor_boxes_3d).float().cuda()
