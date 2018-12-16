# -*- coding:utf-8 -*-
# Author: DuanZhixiang zhixiangduan@deepmotion.ai
# vis box
import os
from PIL import Image, ImageDraw
import math
import numpy as np
from utils.kitti_util import Calibration, Object3d
import sys
import argparse
import cv2
sys.path.append('.')


def parse_kitti_3d(label_path):
    lines = [line.rstrip() for line in open(label_path)]
    objs = [Object3d(line) for line in lines]

    boxes_3d = [obj.box3d for obj in objs]
    points_3ds = []
    for box_3d in boxes_3d:
        [ry, l, h, w, x, y, z] = box_3d
        xmin = x - 1 / 2 * l
        xmax = x + 1 / 2 * l
        ymin = y - 1 / 2 * h
        ymax = y + 1 / 2 * h
        zmin = z - 1 / 2 * w
        zmax = z + 1 / 2 * w
        points_3d = np.stack(
            np.meshgrid([xmin, xmax], [ymin, ymax], [zmin, zmax]), axis=-1)
        points_3d = points_3d.reshape((-1, 3))
        points_3ds.append(points_3d)

    if len(boxes_3d) == 0:
        return np.zeros((0, 8, 3)), np.zeros((0, 7))
    return np.stack(points_3ds, axis=0), np.stack(boxes_3d, axis=0)


def compute_box_3d_in_world(target, p):
    """Takes an target and a project matrix (P) and project the 3D
    bounding box into the world.

    Args:
        target(dict): {'dimension':, 'location':, 'ry':}
        p(numpy.array): 3x4

    Returns:
        (numpy.array): 3x8, coordinates of 8 points in world coordinates.

    """
    rotation_y = target['ry']
    r = [
        math.cos(rotation_y), 0, math.sin(rotation_y), 0, 1, 0,
        -math.sin(rotation_y), 0, math.cos(rotation_y)
    ]
    r = np.array(r).reshape(3, 3)

    l, w, h = target['dimension']

    # The points sequence is 1, 2, 3, 4, 5, 6, 7, 8.
    # Front face: 1, 2, 6, 5; left face: 2, 3, 7, 6
    # Back face: 3, 4, 8, 7; Right face: 4, 1, 5, 8
    x_corners = np.array(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, 0])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h, 0])
    z_corners = np.array(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, 0])

    box_points_coords = np.vstack((x_corners, y_corners, z_corners))
    corners_3d = np.dot(r, box_points_coords)
    corners_3d = corners_3d + np.array(target['location']).reshape(3, 1)

    return corners_3d


def compute_box_3d(target, p, ret_3d=False):
    """Takes an target and a project matrix (P) and project the 3D
    bounding obx into the image plane.

    Args:
        target(dict): {'dimension':, 'location':, 'ry':}
        p(numpy.array): 3x4

    Returns: (numpy.array): 2x9, coordinates of 9 points projected to the plane(including center coord).

    """
    rotation_y = target['ry']
    r = [
        math.cos(rotation_y), 0, math.sin(rotation_y), 0, 1, 0,
        -math.sin(rotation_y), 0, math.cos(rotation_y)
    ]
    r = np.array(r).reshape(3, 3)

    h, w, l = target['dimension']

    # The points sequence is 1, 2, 3, 4, 5, 6, 7, 8.
    # Front face: 1, 2, 6, 5; left face: 2, 3, 7, 6
    # Back face: 3, 4, 8, 7; Right face: 4, 1, 5, 8
    x_corners = np.array(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, 0])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h, 0])
    z_corners = np.array(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, 0])

    box_points_coords = np.vstack((x_corners, y_corners, z_corners))
    corners_3d = np.dot(r, box_points_coords)
    corners_3d = corners_3d + np.array(target['location']).reshape(3, 1)
    corners_3d_homo = np.vstack((corners_3d, np.ones(
        (1, corners_3d.shape[1]))))

    corners_2d = np.dot(p, corners_3d_homo)
    corners_2d_xy = corners_2d[:2, :] / corners_2d[2, :]
    if ret_3d:
        return corners_2d_xy.transpose(), corners_3d

    return corners_2d_xy.transpose()


def load_projection_matrix(calib_file):
    """Load the camera project matrix."""
    assert os.path.isfile(calib_file)
    with open(calib_file) as f:
        lines = f.readlines()
        line = lines[2]
        line = line.split()
        assert line[0] == 'P2:'
        p = [float(x) for x in line[1:]]
        p = np.array(p).reshape(3, 4)
    return p


def draw_boxes(img, box_3d, calib_matrix, save_path, offset=[0, 0], save=True):
    '''
    Args:
        img(PIL.Image):
        box_3d:(np.array): n x 7
        calib_matrix(np.array): 3x3
    '''
    draw = ImageDraw.Draw(img)

    connected_points = [[2, 4, 5], [1, 3, 6], [2, 4, 7], [1, 3, 8], [1, 6, 8],
                        [2, 5, 7], [3, 6, 8], [4, 5, 7]]
    connected_points = np.array(connected_points) - 1
    connected_points = connected_points.tolist()

    for i in range(box_3d.shape[0]):
        target = {}
        target['ry'] = box_3d[i, 0]
        target['dimension'] = box_3d[i, 1:4]
        target['location'] = box_3d[i, 4:]

        corners_xy = compute_box_3d(target, calib_matrix)
        corners_xy[:, 0] -= offset[0]
        corners_xy[:, 1] -= offset[1]
        corners_xy = corners_xy.tolist()
        for i in range(8):
            for j in range(8):
                if j in connected_points[i]:
                    start_point = (corners_xy[i][0], corners_xy[i][1])
                    end_point = (corners_xy[j][0], corners_xy[j][1])
                    draw.line(
                        [start_point, end_point], fill=(255, 170, 30), width=1)

    # if save:
    img.save(save_path)
    img = cv2.imread(save_path)
    cv2.imshow('box_3d', img)
    cv2.waitKey(0)
    # img.show()


def parse_args():
    parser = argparse.ArgumentParser()

    # some key inputs
    parser.add_argument(
        '--kitti',
        dest='kitti',
        help='path to kitti label file or result file',
        type=str)
    parser.add_argument('--img', dest='img', help='path to img', type=str)
    parser.add_argument(
        '--calib', dest='calib', help='path to calib file', type=str)

    # output path
    parser.add_argument(
        '--save_path',
        dest='save_path',
        help='path to output',
        type=str,
        default='./demo.png')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    label_path = args.kitti
    img_path = args.img
    calib_path = args.calib
    save_path = args.save_path

    # load calib matrix(here just P2 is used)
    p2 = load_projection_matrix(calib_path)

    # img
    img = Image.open(img_path)

    # label
    points_3d, boxes_3d = parse_kitti_3d(label_path)

    # final draw it
    draw_boxes(img, boxes_3d, p2, save_path)


if __name__ == '__main__':
    main()
