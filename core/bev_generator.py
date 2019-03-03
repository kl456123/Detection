# -*- coding: utf-8 -*-

from utils import pc_ops
from wavedata.tools.core.voxel_grid_2d import VoxelGrid2D
import numpy as np


class BevGenerator():
    def __init__(self, bev_generator_config):
        self.height_lo = bev_generator_config['height_lo']
        self.height_hi = bev_generator_config['height_hi']
        self.num_slices = bev_generator_config['num_slices']

        self.height_per_division = (
            self.height_hi - self.height_lo) / self.num_slices

        self.voxel_size = bev_generator_config['voxel_size']

    def generate_bev(self, point_cloud, ground_plane, area_extents):
        """
        Slice point cloud to calc the feat of each voxel
        """
        ###############################
        # height maps
        ###############################

        height_maps = []

        for slice_idx in range(self.num_slices):
            height_lo = self.height_lo + slice_idx * self.height_per_division
            height_hi = height_lo + self.height_per_division

            # slice point cloud of being in the specified height interval
            slice_filter = pc_ops.create_slice_filter(
                point_cloud, area_extents, ground_plane, height_lo, height_hi)

            slice_point_cloud = point_cloud[slice_filter]

            if len(slice_point_cloud) > 1:
                voxel_grid_2d = VoxelGrid2D()
                voxel_grid_2d.voxelize_2d(
                    slice_point_cloud,
                    self.voxel_size,
                    extents=area_extents,
                    ground_plane=ground_plane,
                    create_leaf_layout=False)

                voxel_indices = voxel_grid_2d.voxel_indices[:, [0, 2]]

            # (x-z)
            height_map = np.zeros((voxel_grid_2d.num_divisions[0],
                                   voxel_grid_2d.num_divisions[2]))

            voxel_grid_2d.heights -= height_lo
            height_map[voxel_indices[:, 0], voxel_indices[:, 1]] = np.asarray(
                voxel_grid_2d.heights) / self.height_per_division
            height_maps.append(height_map)

        # rotate 90 degrees clock-wise
        height_maps_out = [
            np.flip(
                height_maps[map_idx].transpose(), axis=0)
            for map_idx in range(len(height_maps))
        ]

        #############################
        # density map
        #############################
        # import ipdb
        # ipdb.set_trace()
        density_slice_filter = pc_ops.create_slice_filter(
            point_cloud, area_extents, ground_plane, self.height_lo,
            self.height_hi)
        density_points = point_cloud[density_slice_filter]

        density_voxel_grid_2d = VoxelGrid2D()
        density_voxel_grid_2d.voxelize_2d(
            density_points,
            self.voxel_size,
            extents=area_extents,
            ground_plane=ground_plane,
            create_leaf_layout=False)

        density_voxel_indices = density_voxel_grid_2d.voxel_indices[:, [0, 2]]
        density_map = np.zeros((density_voxel_grid_2d.num_divisions[0],
                                density_voxel_grid_2d.num_divisions[2]))
        density_value = self.calc_density(
            density_voxel_grid_2d.num_pts_in_voxel)
        density_map[density_voxel_indices[:, 0],
                    density_voxel_indices[:, 1]] = density_value

        density_map = np.flip(density_map.transpose(), axis=0)

        bev_maps = dict()
        bev_maps['height_maps'] = height_maps_out
        bev_maps['density_map'] = density_map
        return bev_maps

    def calc_density(self, num_pts_per_voxel, norm_value=16):
        # Density is calculated as min(1.0, log(N+1)/log(x))
        # x=64 for stereo, x=16 for lidar, x=64 for depth
        return np.minimum(1.0, np.log(num_pts_per_voxel + 1) / norm_value)
