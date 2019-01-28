# -*- coding: utf-8 -*-

import torch
import core.ops as ops


class VoxelGenerator(object):
    def __init__(self, voxel_generator_config):
        self.voxel_size = voxel_generator_config['voxel_size']
        # WHD(num)
        # self.lattice_dims = voxel_generator_config['lattice_dims']
        # xyz(meters)
        self.grid_dims = voxel_generator_config['grid_dims']

        self.voxels = None

        # from bottom to top
        self.high_interval = voxel_generator_config['high_interval']

        self.y0 = voxel_generator_config['ground_plane']

    def generate_voxels(self):
        """
        generate all voxels in ground plane
        """
        lattice_dims = self.grid_dims / self.voxel_size
        x_inds = torch.arange(0, lattice_dims[0]).cuda()
        y_inds = torch.arange(0, lattice_dims[1]).cuda()
        z_inds = torch.arange(0, lattice_dims[2]).cuda()
        x_inds, y_inds = ops.meshgrid(x_inds, y_inds)
        x_inds, z_inds1 = ops.meshgrid(x_inds, z_inds)
        y_inds, z_inds2 = ops.meshgrid(y_inds, z_inds)
        z_inds = z_inds1

        corner_coords = torch.stack([x_inds, y_inds, z_inds], dim=-1).float()
        corner_coords *= self.voxel_size

        center_offset = []
        center_coords = corner_coords + center_offset

        original_offset = []
        center_coords = center_coords + original_offset

        return center_coords

    def proj_voxels_3dTo2d(self, p2):
        """
        Project bbox in 3d to bbox in 2d
        """

