# -*- coding: utf-8 -*-

from utils import geometry_utils


def encode_lines(lines, proposals):
    """
    Args:
        lines: shape(N, 2, 2)
    """
    proposals_xywh = geometry_utils.torch_xyxy_to_xywh(
        proposals.unsqueeze(0))[0]
    encoded_lines = (
        lines - proposals_xywh[:, None, :2]) / proposals_xywh[:, None, 2:]
    return encoded_lines


def encode_points(points, proposals):
    """
    Args:
        points: shape(N, 2)
        proposals: shape(N, 4)
    """
    proposals_xywh = geometry_utils.torch_xyxy_to_xywh(
        proposals.unsqueeze(0))[0]
    encoded_points = (points - proposals_xywh[:, :2]) / proposals_xywh[:, 2:]
    return encoded_points


def decode_points(encoded_points, proposals):
    proposals_xywh = geometry_utils.torch_xyxy_to_xywh(
        proposals.unsqueeze(0))[0]
    points = encoded_points * proposals_xywh[:, 2:] + proposals_xywh[:, :2]
    return points


def decode_lines(encoded_lines, proposals):
    proposals_xywh = geometry_utils.torch_xyxy_to_xywh(
        proposals.unsqueeze(0))[0]
    lines = encoded_lines.view(
        -1, 2, 2) * proposals_xywh[:, None, 2:] + proposals_xywh[:, None, :2]
    return lines.view(-1, 4)


def decode_ry(encoded_lines, proposals, p2):
    lines = decode_lines(encoded_lines, proposals)

    ry = geometry_utils.torch_pts_2d_to_dir_3d(
        lines.unsqueeze(0), p2.unsqueeze(0))[0].unsqueeze(-1)
    return ry
