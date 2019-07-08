# -*- coding: utf-8 -*-
import numpy as np
import math

from utils import geometry_utils


def generate_mid_line(boxes_3d, p2):
    """
    Args:
        boxes_3d: shape(7), (xyz,hwl, ry)
    """
    boxes_3d = np.asarray(boxes_3d).reshape(1, -1)
    corners_3d = geometry_utils.boxes_3d_to_corners_3d(boxes_3d)

    # find the nearest line
    dist = np.linalg.norm(corners_3d, axis=-1)
    argmin = dist.argmin(axis=-1)
    corners_2d = geometry_utils.boxes_3d_to_corners_2d(boxes_3d, p2)
    row = np.arange(corners_2d.shape[0])
    mid_line_2d_tmp = corners_2d[row, argmin]
    return mid_line_2d_tmp[0]


def find_best_ry(boxes_3d, mid_line_2d_x, p2, possible_rys=[0, 0.5 * np.pi]):
    """
    Args:
        boxes_3d: shape(7), (xyz,hwl)
    """
    # import ipdb
    # ipdb.set_trace()
    res = []
    for ry in possible_rys:
        boxes_3d_ry = boxes_3d + [ry]
        mid_line_2d_tmp = generate_mid_line(boxes_3d_ry, p2)

        # compare mid_line_2d_tmp with the given mid_line
        # the nearest ,the better
        res.append(np.abs(mid_line_2d_tmp[0] - mid_line_2d_x))

    res = np.asarray(res)
    argmin = res.argmin(axis=-1)
    return possible_rys[argmin]


def main():
    # use 000001.png for testing
    # label for 000001

    boxes_3d_gt = [-16.53, 2.39, 58.49, 1.67, 1.87, 3.69, 1.57]
    p2 = np.asarray([
        7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02,
        4.485728000000e+01, 0.000000000000e+00, 7.215377000000e+02,
        1.728540000000e+02, 2.163791000000e-01, 0.000000000000e+00,
        0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03
    ]).reshape(3, 4)
    # no ry output
    boxes_3d_pred = [-16.53, 2.39, 58.49, 1.67, 1.87, 3.69]
    mid_line_gt = generate_mid_line(boxes_3d_gt, p2)
    ry = find_best_ry(boxes_3d_pred, mid_line_gt[0], p2)
    print('(pred/gt)({}/{})'.format(ry, boxes_3d_gt[-1]))


if __name__ == '__main__':
    main()
