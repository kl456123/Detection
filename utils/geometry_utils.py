# -*- coding: utf-8 -*-
import numpy as np
import torch


##########################
# some common np ops
##########################
def bmm(a, b):
    """
    Args:
        a: shape(N, M, K)
        b: shape(N, K, S)
    """
    b_T = b.transpose(0, 2, 1)
    return (a[:, :, None, :] * b_T[:, None, :, :]).sum(axis=-1)


def ry_to_rotation_matix(rotation_y):
    zeros = np.zeros_like(rotation_y)
    ones = np.ones_like(rotation_y)
    rotation_matrix = np.asarray([
        np.cos(rotation_y), zeros, np.sin(rotation_y), zeros, ones, zeros,
        -np.sin(rotation_y), zeros, np.cos(rotation_y)
    ]).reshape(-1, 3, 3)
    return rotation_matrix


def torch_ry_to_rotation_matix(rotation_y):
    zeros = torch.zeros_like(rotation_y).type_as(rotation_y)
    ones = torch.ones_like(rotation_y).type_as(rotation_y)
    rotation_matrix = torch.stack(
        [
            torch.cos(rotation_y), zeros, torch.sin(rotation_y), zeros, ones,
            zeros, -torch.sin(rotation_y), zeros, torch.cos(rotation_y)
        ],
        dim=-1).reshape(-1, 3, 3)
    return rotation_matrix


def boxes_3d_to_corners_3d(boxes):
    """
    Args:
    boxes: shape(N, 7), (xyz,lhw, ry)
    corners_3d: shape()
    """
    ry = boxes[:, -1]
    location = boxes[:, :3]
    l = boxes[:, 3]
    h = boxes[:, 4]
    w = boxes[:, 5]
    zeros = np.zeros_like(l)
    rotation_matrix = ry_to_rotation_matix(ry)

    x_corners = np.array(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    y_corners = np.array([zeros, zeros, zeros, zeros, -h, -h, -h, -h])
    z_corners = np.array(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2])

    # shape(N, 3, 8)
    box_points_coords = np.stack((x_corners, y_corners, z_corners), axis=0)
    # rotate and translate
    # shape(N, 3, 8)
    corners_3d = bmm(rotation_matrix, box_points_coords.transpose(2, 0, 1))
    corners_3d = corners_3d + location[..., None]
    # shape(N, 8, 3)
    return corners_3d.transpose(0, 2, 1)
    # shape(N, 4, 8)
    # corners_3d_homo = np.concatenate(


# (corners_3d, np.ones_like(corners_3d[:, -1:, :])), axis=1)


def points_3d_to_points_2d(points_3d, p2):
    """
    Args:
        points_3d: shape(N, 3)
        p2: shape(3,4)
    Returns:
        points_2d: shape(N, 2)
    """

    # import ipdb
    # ipdb.set_trace()
    points_3d_homo = np.concatenate(
        (points_3d, np.ones_like(points_3d[:, -1:])), axis=-1)
    points_2d_homo = np.dot(p2, points_3d_homo.T).T
    points_2d_homo /= points_2d_homo[:, -1:]
    return points_2d_homo[:, :2]


def boxes_3d_to_corners_2d(boxes, p2):
    """
    Args:
    boxes: shape(N, 7)
    corners_2d: shape(N, )
    """
    corners_3d = boxes_3d_to_corners_3d(boxes)
    corners_2d = points_3d_to_points_2d(corners_3d.reshape((-1, 3)),
                                        p2).reshape(-1, 8, 2)
    return corners_2d


def corners_2d_to_boxes_2d(corners_2d):
    """
    Find the closure
    Args:
        corners_2d: shape(N, 8, 2)
    """
    xmin = corners_2d[:, :, 0].min(axis=-1)
    xmax = corners_2d[:, :, 0].max(axis=-1)
    ymin = corners_2d[:, :, 1].min(axis=-1)
    ymax = corners_2d[:, :, 1].max(axis=-1)

    return np.stack([xmin, ymin, xmax, ymax], axis=-1)


###########################
# pytorch
###########################
def torch_boxes_3d_to_corners_3d(boxes):
    """
    Args:
        boxes: shape(N, 7), (xyz,lhw, ry)
        corners_3d: shape()
    """
    ry = boxes[:, -1]
    location = boxes[:, :3]
    l = boxes[:, 3]
    h = boxes[:, 4]
    w = boxes[:, 5]
    zeros = torch.zeros_like(l).type_as(l)
    rotation_matrix = torch_ry_to_rotation_matix(ry)

    x_corners = torch.stack(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], dim=0)
    y_corners = torch.stack(
        [zeros, zeros, zeros, zeros, -h, -h, -h, -h], dim=0)
    z_corners = torch.stack(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], dim=0)

    # shape(N, 3, 8)
    box_points_coords = torch.stack((x_corners, y_corners, z_corners), dim=0)
    # rotate and translate
    # shape(N, 3, 8)
    corners_3d = torch.bmm(rotation_matrix,
                           box_points_coords.transpose(2, 0, 1))
    corners_3d = corners_3d + location.unsqueeze(-1)
    # shape(N, 8, 3)
    return corners_3d.permute(0, 2, 1)
