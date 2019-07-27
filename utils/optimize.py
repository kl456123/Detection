# -*- coding: utf-8 -*-
"""
How to according to given information to generate or refine 3d bbox
Dims and depth can be predicted well(prior), just optimize xy and ry
"""

import sys
import numpy as np
import os
from utils import geometry_utils
from scipy import optimize
from data import datasets
from core import constants
from utils.drawer import ImageVisualizer


def build_dataset(dataset_type='kitti'):
    if dataset_type == 'kitti':
        dataset_config = {
            "type": "mono_3d_kitti",
            "classes": ["Car", "Pedestrian"],
            "cache_bev": False,
            "dataset_file": "data/demo.txt",
            "root_path": "/data"
        }
    elif dataset_type == 'nuscenes':
        dataset_config = {
            'type': 'nuscenes',
            'root_path': '/data/nuscenes',
            'dataset_file': 'trainval.json',
            'data_path': 'samples/CAM_FRONT',
            'label_path': '.',
            'classes': ['car', 'pedestrian', 'truck']
        }
    elif dataset_type == 'keypoint_kitti':
        dataset_config = {
            "type": "keypoint_kitti",
            "classes": ["Car", "Pedestrian"],
            "cache_bev": False,
            "dataset_file": "data/demo.txt",
            "root_path": "/data"
        }

    dataset = datasets.build(dataset_config, transform=None, training=True)

    return dataset


def optimizer_wrapper(f):
    def new_f(a):
        return optimize.minimize(f, a)

    return new_f


def optimize_ry(xy, dims, p2, init_ry, depth, corners_2d, p=2):
    @optimizer_wrapper
    def func_given_depth(X):
        """
        Args:
            X: (x, y, ry, dims)
        """
        # import ipdb
        # ipdb.set_trace()
        # eight equations
        ry = X[0]
        h = X[1]
        w = X[2]
        l = X[3]
        # h,w,l = dims

        boxes_3d = np.asarray([xy[0], xy[1], depth, h, w, l, ry]).reshape(
            1, -1)
        corners_2d_preds = geometry_utils.boxes_3d_to_corners_2d(boxes_3d, p2)
        dist = corners_2d.reshape(-1) - corners_2d_preds.reshape(-1)
        dist = np.linalg.norm(dist, p)
        return dist

    init_value = np.asarray([init_ry, dims[0], dims[1], dims[2]])
    return func_given_depth(init_value)


def optimize_xy(init_xy, init_dims, p2, ry, depth, corners_2d, p=2):
    @optimizer_wrapper
    def func_given_depth(X):
        """
        Args:
            X: (x, y, ry, dims)
        """
        # import ipdb
        # ipdb.set_trace()
        # eight equations
        x = X[0]
        y = X[1]
        h = X[2]
        w = X[3]
        l = X[4]

        boxes_3d = np.asarray([x, y, depth, h, w, l, ry]).reshape(1, -1)
        corners_2d_preds = geometry_utils.boxes_3d_to_corners_2d(boxes_3d, p2)
        dist = corners_2d.reshape(-1) - corners_2d_preds.reshape(-1)
        dist = np.linalg.norm(dist, p)
        return dist

    init_value = np.asarray(
        [init_xy[0], init_xy[1], init_dims[0], init_dims[1], init_dims[2]])
    return func_given_depth(init_value)


def optimize_corners_2d(corners_2d,
                        dims,
                        p2,
                        ground_plane=None,
                        depth=None,
                        p=2,
                        init_center_2d=None,
                        init_ry=None):
    """
    Args:
        corners_2d: shape(8, 2)
        depth: shape(1)
        dims: shape(3)
        ground_plane: shape(4) can be None
    Returns:
        X: shape(3) (x, y, ry)
    """
    use_depth = True if depth is not None else False

    @optimizer_wrapper
    def func_given_depth(X):
        """
        Args:
            X: (x, y, ry)
        """
        # import ipdb
        # ipdb.set_trace()
        # eight equations
        x = X[0]
        y = X[1]
        # x = init_center_2d[0]
        # y = init_center_2d[1]
        if init_ry is not None:
            ry = init_ry
        depth_deltas = X[3]
        dims_scale = X[4]
        dims_rescale = dims
        depth_refine = depth * dims_scale
        boxes_3d = np.asarray([
            x, y, depth_refine, dims_rescale[0], dims_rescale[1],
            dims_rescale[2], ry
        ]).reshape(1, -1)
        corners_2d_preds = geometry_utils.boxes_3d_to_corners_2d(boxes_3d, p2)
        dist = corners_2d.reshape(-1) - corners_2d_preds.reshape(-1)
        dist = np.linalg.norm(dist, p)
        return dist

    @optimizer_wrapper
    def func_given_ground_plane(X):
        """
        Args:
            X: (x, y, ry)
        """
        # import ipdb
        # ipdb.set_trace()
        # eight equations
        x = X[0]
        y = X[1]
        ry = X[2]
        boxes_3d = np.asarray([x, y, depth, dims[0], dims[1], dims[2],
                               ry]).reshape(1, -1)
        corners_2d_preds = geometry_utils.boxes_3d_to_corners_2d(boxes_3d, p2)
        dist = corners_2d.reshape(-1) - corners_2d_preds.reshape(-1)
        dist = np.linalg.norm(dist, p)
        return dist

    f = func_given_depth if use_depth else func_given_ground_plane

    depth_deltas_init = 0
    dims_scale_init = 1
    if init_center_2d is None:
        init_center_2d = [0, 0]
    init_value = np.asarray([
        init_center_2d[0], init_center_2d[1], -1.57, depth_deltas_init,
        dims_scale_init
    ])
    return f(init_value)


def read_txt(label_path):
    """
    x1, y1, x2, y2
    """
    num_cols = 4 + 8 * 2 + 3 + 1 + 1
    labels = np.loadtxt(label_path).reshape(-1, num_cols)

    # info = {
    # 'corners': corners_2d,
    # 'dims': dims,
    # 'depth': depth,
    # 'boxes_2d': boxes_2d
    # }
    # return info
    return labels


def convert_to_corners_3d(dims, depth, xyry):
    # import ipdb
    # ipdb.set_trace()
    x, y, ry = xyry[:3]
    dims_scale = xyry[-1]
    dims = dims * dims_scale
    label_boxes_3d = np.asarray([x, y, depth, dims[0], dims[1], dims[2],
                                 ry]).reshape(-1, 7)
    corners_3d = geometry_utils.boxes_3d_to_corners_3d(label_boxes_3d)
    return corners_3d


def generate_visualizer():
    image_dir = '/data/object/training/image_2'
    result_dir = './results/data'
    save_dir = 'results/optimized'
    calib_dir = '/data/object/training/calib'
    label_dir = None
    #  label_dir = '/data/object/training/label_2'
    calib_file = None
    visualizer = ImageVisualizer(
        image_dir,
        result_dir,
        label_dir=label_dir,
        calib_dir=calib_dir,
        calib_file=calib_file,
        online=False,
        save_dir=save_dir)
    return visualizer


def estimate_ry(corners_2d, p2):
    """
    Args:
        corners_2d: shape(N, 8, 2)
        p2: shape(3, 4)
    """
    corners_2d = corners_2d.reshape(-1, 8, 2)
    front_points = corners_2d[:, [0]].mean(axis=1)
    rear_points = corners_2d[:, [3]].mean(axis=1)
    lines = np.concatenate([rear_points, front_points], axis=-1)[None]
    ry = geometry_utils.pts_2d_to_dir_3d(lines, p2[None])[0][..., None]
    return ry


def estimate_ry_v2(corners_2d, p2):
    """
    Args:
        corners_2d: shape(N, 8, 2)
        p2: shape(3, 4)
    """
    corners_2d = corners_2d.reshape(-1, 8, 2)
    # front_points = corners_2d[:, [0, 1, 5, 4]].mean(axis=1)
    # rear_points = corners_2d[:, [2, 3, 6, 7]].mean(axis=1)
    side_lines = corners_2d[:, [0, 3, 1, 2, 4, 7, 5, 6]]
    front_lines = corners_2d[:, [0, 1, 2, 3, 4, 5, 6, 7]]

    side_lines = side_lines.reshape(-1, 4, 4)
    front_lines = front_lines.reshape(-1, 4, 4)
    ry = geometry_utils.pts_2d_to_dir_3d(side_lines, p2[None])
    return ry.mean(axis=-1, keepdims=True)


def estimate_location(corners_2d, depth, p2):
    """
        center of bottom face of 3d bbox
    """
    corners_3d = geometry_utils.points_2d_to_points_3d(
        corners_2d.reshape(-1, 2), depth, p2)
    bottom_corners = corners_3d[:, [0, 1, 2, 3]]

    location = bottom_corners.mean(axis=1)
    return location


def mainv2():
    """
    self optimization
    """
    label_dir = 'results/data'
    corners_label_dir = 'results/corners'
    p2 = np.asarray([
        7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02,
        4.575831000000e+01, 0.000000000000e+00, 7.070493000000e+02,
        1.805066000000e+02, -3.454157000000e-01, 0.000000000000e+00,
        0.000000000000e+00, 1.000000000000e+00, 4.981016000000e-03
    ]).reshape(3, 4)
    visualizer = generate_visualizer()
    image_dir = '/data/object/training/image_2'
    for sample_ind, sample_name in enumerate(
            sorted(os.listdir(corners_label_dir))):
        # sample_name = '000006.txt'
        corners_label_path = os.path.join(corners_label_dir, sample_name)
        labels = read_txt(corners_label_path)
        num_instances = labels.shape[0]
        boxes_2d = labels[:, :4]
        corners_2d = labels[:, 4:4 + 16]
        dims = labels[:, 4 + 16:4 + 16 + 3]
        depth = labels[:, 4 + 16 + 3:4 + 16 + 3 + 1]
        conf = labels[:, -1:]
        ground_plane = None
        center_2d = (boxes_2d[:, :2] + boxes_2d[:, 2:]) / 2
        points_3d = geometry_utils.points_2d_to_points_3d(center_2d, depth, p2)
        center_2d = points_3d[:, :2]
        # import ipdb
        # ipdb.set_trace()
        ry = estimate_ry(corners_2d, p2)
        # location = estimate_location(corners_2d, depth, p2)

        # optimize xy
        optimized_xys = []
        for ind in range(num_instances):
            optimized_xys.append(
                optimize_xy(
                    center_2d[ind], [1, 1, 1],
                    p2,
                    ry[ind, 0],
                    depth[ind, 0],
                    corners_2d[ind],
                    p=2).x[:2])
        optimized_xys = np.asarray(optimized_xys).reshape(-1, 2)

        # optimize rys
        # import ipdb
        # ipdb.set_trace()
        optimized_rys = []
        for ind in range(num_instances):
            optimized_rys.append(
                optimize_ry(
                    optimized_xys[ind],
                    dims[ind],
                    p2,
                    ry[ind, 0],
                    depth[ind, 0],
                    corners_2d[ind],
                    p=1).x[0])
        optimized_rys = np.asarray(optimized_rys).reshape(-1, 1)

        boxes_3d = np.concatenate(
            [optimized_xys[:, :2], depth, dims, ry], axis=-1)
        corners_3d = geometry_utils.boxes_3d_to_corners_3d(boxes_3d)

        # corners_3d = np.concatenate(corners_3d, axis=0)
        image_path = os.path.join(image_dir, '{}.png'.format(sample_name[:-4]))
        visualizer.render_image_corners_2d(
            image_path,
            corners_2d=None,
            boxes_2d=None,
            corners_3d=corners_3d,
            p2=p2)

        dets = np.concatenate([boxes_2d, boxes_3d, conf], axis=-1)
        # save to txt
        label_path = os.path.join(label_dir, '{}.txt'.format(sample_name[:-4]))
        save_dets(dets, label_path)
        sys.stdout.write('\rind: {} sample_name: {}'.format(
            sample_ind, sample_name))
        sys.stdout.flush()


def save_dets(dets, label_path):
    res_str = []
    kitti_template = 'Car -1 -1 -10 {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.8f}'
    with open(label_path, 'w') as f:
        for det in dets:
            # xmin, ymin, xmax, ymax, cf, l, h, w, ry, x, y, z = det
            xmin, ymin, xmax, ymax, x, y, z, h, w, l, ry, cf = det
            res_str.append(
                kitti_template.format(xmin, ymin, xmax, ymax, h, w, l, x, y, z,
                                      ry, cf))
        f.write('\n'.join(res_str))


def main():

    label_dir = 'results/data'
    p2 = np.asarray([
        7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02,
        4.575831000000e+01, 0.000000000000e+00, 7.070493000000e+02,
        1.805066000000e+02, -3.454157000000e-01, 0.000000000000e+00,
        0.000000000000e+00, 1.000000000000e+00, 4.981016000000e-03
    ]).reshape(3, 4)
    visualizer = generate_visualizer()
    image_dir = '/data/object/training/image_2'
    for sample_name in os.listdir(label_dir):
        label_path = os.path.join(label_dir, sample_name)
        labels = read_txt(label_path)
        num_instances = labels.shape[0]
        boxes_2d = labels[:, :4]
        corners_2d = labels[:, 4:4 + 16]
        dims = labels[:, 4 + 16:4 + 16 + 3]
        depth = labels[:, -2:-1]
        conf = labels[:, -1:]
        ground_plane = None
        center_2d = (boxes_2d[:, :2] + boxes_2d[:, 2:]) / 2
        points_3d = geometry_utils.points_2d_to_points_3d(center_2d, depth, p2)
        center_2d = points_3d[:, :2]

        corners_3d = []
        for ind in range(num_instances):
            res = optimize_corners_2d(
                corners_2d[ind],
                dims[ind],
                p2,
                ground_plane=ground_plane,
                depth=depth[ind],
                init_center_2d=center_2d[ind],
                p=2)
            # import ipdb
            # ipdb.set_trace()
            if not res.success:
                print('{} fail'.format(ind))
                continue
            if res.fun > 100:
                print('{} fail'.format(ind))
                continue
            corners_3d.append(
                convert_to_corners_3d(dims[ind], depth[ind], res.x))
        corners_3d = np.concatenate(corners_3d, axis=0)
        image_path = os.path.join(image_dir, '{}.png'.format(sample_name[:-4]))
        visualizer.render_image_corners_2d(
            image_path,
            corners_2d=None,
            boxes_2d=None,
            corners_3d=corners_3d,
            p2=p2)


def test():
    corners_2d = []
    dims = np.asarray([1.52563191462, 1.62856739989, 3.88311640418])
    ground_plane = None
    #  depth = []
    p2 = np.asarray([
        7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02,
        4.575831000000e+01, 0.000000000000e+00, 7.070493000000e+02,
        1.805066000000e+02, -3.454157000000e-01, 0.000000000000e+00,
        0.000000000000e+00, 1.000000000000e+00, 4.981016000000e-03
    ]).reshape(3, 4)

    # import ipdb
    # ipdb.set_trace()
    dataset = build_dataset()
    for sample in dataset:
        label_boxes_3d = sample[constants.KEY_LABEL_BOXES_3D]
        num_instances = sample[constants.KEY_NUM_INSTANCES]
        label_boxes_3d = label_boxes_3d[:num_instances]
        corners_2d = geometry_utils.boxes_3d_to_corners_2d(label_boxes_3d, p2)
        depth = label_boxes_3d[:, 2]
        res = optimize_corners_2d(
            corners_2d[0], dims, p2, ground_plane=ground_plane, depth=depth[0])
        print(res)


if __name__ == '__main__':
    mainv2()
