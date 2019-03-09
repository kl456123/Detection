# -*- coding: utf-8 -*-
"""
Utils for processing point cloud
"""
import numpy as np
from wavedata.tools.core.integral_image_2d import IntegralImage2D
from wavedata.tools.core.voxel_grid_2d import VoxelGrid2D

import torch


def create_slice_filter(point_cloud,
                        extents,
                        ground_plane,
                        height_lo=0.2,
                        height_hi=2.0):
    offset_filter = get_point_filter(point_cloud, extents, ground_plane,
                                     height_hi)

    road_filter = get_point_filter(point_cloud, extents, ground_plane,
                                   height_lo)

    sliced_filter = np.logical_xor(offset_filter, road_filter)

    return sliced_filter
    # return filtered_point_cloud


def create_sliced_voxel_grid_2d(point_cloud,
                                extents,
                                voxel_size,
                                ground_plane,
                                height_lo=0.2,
                                height_hi=2.0):
    sliced_filter = create_slice_filter(point_cloud, extents, ground_plane,
                                        height_lo, height_hi)
    filtered_points = point_cloud[sliced_filter]
    # Create Voxel Grid
    voxel_grid_2d = VoxelGrid2D()
    voxel_grid_2d.voxelize_2d(
        filtered_points,
        voxel_size,
        extents=extents,
        ground_plane=ground_plane,
        create_leaf_layout=True)

    return voxel_grid_2d


def get_point_filter(point_cloud, extents, ground_plane=None, offset_dist=2.0):
    """
    Creates a point filter using the 3D extents and ground plane

    :param point_cloud: Point cloud in the form [[x,...],[y,...],[z,...]]
    :param extents: 3D area in the form
        [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
    :param ground_plane: Optional, coefficients of the ground plane
        (a, b, c, d)
    :param offset_dist: If ground_plane is provided, removes points above
        this offset from the ground_plane
    :return: A binary mask for points within the extents and offset plane
    """
    point_cloud = np.transpose(point_cloud)
    point_cloud = np.asarray(point_cloud)

    # Filter points within certain xyz range
    x_extents = extents[0]
    y_extents = extents[1]
    z_extents = extents[2]

    extents_filter = (point_cloud[0] > x_extents[0]) & \
                    (point_cloud[0] < x_extents[1]) & \
                    (point_cloud[1] > y_extents[0]) & \
                    (point_cloud[1] < y_extents[1]) & \
                    (point_cloud[2] > z_extents[0]) & \
                    (point_cloud[2] < z_extents[1])

    if ground_plane is not None:
        ground_plane = np.array(ground_plane)

        # Calculate filter using ground plane
        ones_col = np.ones(point_cloud.shape[1])
        padded_points = np.vstack([point_cloud, ones_col])

        offset_plane = ground_plane + [0, 0, 0, -offset_dist]

        offset_plane = ground_plane + [0, 0, 0, -offset_dist]

        # Create plane filter
        dot_prod = np.dot(offset_plane, padded_points)
        plane_filter = dot_prod < 0

        # Combine the two filters
        point_filter = np.logical_and(extents_filter, plane_filter)
    else:
        # Only use the extents for filtering
        point_filter = extents_filter

    return point_filter


def get_empty_anchor_filter_2d(anchors, voxel_grid_2d, density_threshold=1):
    """ Returns a filter for empty anchors from the given 2D anchor list

    Args:
        anchors: list of 3d anchors in the format
            N x [x, y, z, dim_x, dim_y, dim_z]
        voxel_grid_2d: a VoxelGrid object containing a 2D voxel grid of
            point cloud used to filter the anchors
        density_threshold: minimum number of points in voxel to keep the anchor

    Returns:
        anchor filter: N Boolean mask
    """
    # Remove y dimensions from anchors to project into BEV
    anchors_2d = anchors[:, [0, 2, 3, 5]]

    # Get Integral image of the voxel, add 1 since filled = 0, empty is -1
    leaf_layout = voxel_grid_2d.leaf_layout_2d + 1
    leaf_layout = np.squeeze(leaf_layout)
    integral_image = IntegralImage2D(leaf_layout)

    # Make anchor container
    anchor_container = np.zeros([len(anchors_2d), 4]).astype(np.uint32)

    num_anchors = len(anchors_2d)

    # Set up objects containing corners of anchors
    top_left_up = np.zeros([num_anchors, 2]).astype(np.float32)
    bot_right_down = np.zeros([num_anchors, 2]).astype(np.float32)

    # Calculate minimum corner
    top_left_up[:, 0] = anchors_2d[:, 0] - (anchors_2d[:, 2] / 2.)
    top_left_up[:, 1] = anchors_2d[:, 1] - (anchors_2d[:, 3] / 2.)

    # Calculate maximum corner
    bot_right_down[:, 0] = anchors_2d[:, 0] + (anchors_2d[:, 2] / 2.)
    bot_right_down[:, 1] = anchors_2d[:, 1] + (anchors_2d[:, 3] / 2.)

    # map_to_index() expects N x 2 points
    anchor_container[:, :2] = voxel_grid_2d.map_to_index(top_left_up)
    anchor_container[:, 2:] = voxel_grid_2d.map_to_index(bot_right_down)

    # Transpose to pass into query()
    anchor_container = anchor_container.T

    # Get point density score for each anchor
    point_density_score = integral_image.query(anchor_container)

    point_density_score = torch.tensor(point_density_score).float().cuda()
    # Create the filter
    anchor_filter = point_density_score >= density_threshold

    return anchor_filter
