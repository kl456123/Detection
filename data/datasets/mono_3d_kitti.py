# -*- coding: utf-8 -*-

from data.datasets.kitti import KITTIDataset
from wavedata.tools.obj_detection import obj_utils
from core import constants
from utils import geometry_utils
import numpy as np
from utils.registry import DATASETS
from utils.box_vis import compute_box_3d
import torch
from utils import image_utils

from utils import pointcloud_utils
from PIL import Image
import cv2

MEAN_DIMS = {
    # KITTI
    'Car': [3.88311640418, 1.62856739989, 1.52563191462],
    'Van': [5.06763659, 1.9007158, 2.20532825],
    'Truck': [10.13586957, 2.58549199, 3.2520595],
    'Pedestrian': [0.84422524, 0.66068622, 1.76255119],
    'Person_sitting': [0.80057803, 0.5983815, 1.27450867],
    'Cyclist': [1.76282397, 0.59706367, 1.73698127],
    'Tram': [16.17150617, 2.53246914, 3.53079012],
    'Misc': [3.64300781, 1.54298177, 1.92320313],

    # NUSCENES
    'car': [3.88311640418, 1.62856739989, 1.52563191462],
    'bus': [3.88311640418, 1.62856739989, 1.52563191462],
    'truck': [10.13586957, 2.58549199, 3.2520595],
    'pedestrian': [0.84422524, 0.66068622, 1.76255119],
    'bicycle': [1.76282397, 0.59706367, 1.73698127],
    'motorcycle': [1.76282397, 0.59706367, 1.73698127],
    'trailer': [10.13586957, 2.58549199, 3.2520595],
    'construction_vehicle': [10.13586957, 2.58549199, 3.2520595],
}


def modify_cls_orient(cls_orient, left_side, right_side):
    """
    For special case, classifiy it from common case
    """
    left_dir = (left_side[0] - left_side[1])
    right_dir = (right_side[0] - right_side[1])
    cond = left_dir[0] * right_dir[0] < 0
    if cond:
        return 2
    else:
        return cls_orient


def truncate_box(box_2d, line, normalize=True):
    """
    Args:
        dims_2d:
        line:
    Return: cls_orient:
            reg_orient:
    """
    # import ipdb
    # ipdb.set_trace()
    direction = (line[0] - line[1])
    if direction[0] * direction[1] == 0:
        cls_orient = -1
    else:
        cls_orient = direction[1] / direction[0] > 0
        cls_orient = cls_orient.astype(np.int32)
    # cls_orient = direction[0] > 0
    reg_orient = np.abs(direction)

    # normalize
    if normalize:
        # w, h = dims_2d
        h = box_2d[3] - box_2d[1] + 1
        w = box_2d[2] - box_2d[0] + 1

        reg_orient[0] /= w
        reg_orient[1] /= h
        # reg_orient = np.log(reg_orient)
    return cls_orient, reg_orient


@DATASETS.register('mono_3d_kitti')
@DATASETS.register('nuscenes_kitti')
class Mono3DKITTIDataset(KITTIDataset):
    def __init__(self, config, transform=None, training=True, logger=None):
        super().__init__(config, transform, training, logger)
        self.use_proj_2d = config.get('use_proj_2d', False)

    def _generate_mean_dims(self):
        mean_dims = []
        for class_type in self.classes[1:]:
            mean_dims.append(MEAN_DIMS[class_type][::-1])
        return np.stack(mean_dims, axis=0).astype(np.float32)

    def get_sample(self, index):
        sample = super().get_sample(index)
        # image
        mean_dims = self._generate_mean_dims()
        sample[constants.KEY_MEAN_DIMS] = mean_dims

        if self.training and self.use_proj_2d:
            # use boxes_3d_proj rather than boxes 2d
            label_boxes_3d = sample[constants.KEY_LABEL_BOXES_3D]
            p2 = sample[constants.KEY_STEREO_CALIB_P2]
            boxes_3d_proj = geometry_utils.boxes_3d_to_boxes_2d(
                label_boxes_3d, p2)
            sample[constants.KEY_LABEL_BOXES_2D] = boxes_3d_proj

        if not self.training:
            sample[constants.KEY_STEREO_CALIB_P2_ORIG] = np.copy(
                sample[constants.KEY_STEREO_CALIB_P2])

        # if self.training:
        # sample[constants.KEY_LABEL_POINTCLOUDS] = pc

        return sample

    def pad_sample(self, sample):
        sample = super().pad_sample(sample)
        image_path = sample[constants.KEY_IMAGE_PATH]
        sample_name = self.get_sample_name_from_path(image_path)
        pc = obj_utils.get_lidar_point_cloud(
            int(sample_name),
            self.calib_dir,
            self.velo_dir,
            im_size=None,
            min_intensity=None)
        p2 = sample[constants.KEY_STEREO_CALIB_P2]
        h, w = sample[constants.KEY_IMAGE_INFO].astype(np.int)[:2]

        label_boxes_3d = sample[constants.KEY_LABEL_BOXES_3D]
        num_instances = sample[constants.KEY_NUM_INSTANCES]
        depth = np.zeros((h, w), dtype=np.float32)
        # 0 refers to bg
        mask = np.zeros((h, w), dtype=np.int64)

        for i in range(num_instances):
            label_box_3d = label_boxes_3d[i]
            instance_pc = pointcloud_utils.box_3d_filter(label_box_3d, pc[:3])

            # draw depth
            instance_depth = pointcloud_utils.cam_pts_to_rect_depth(
                instance_pc[:3, :], K=p2[:3, :3], h=h, w=w)
            mask[instance_depth > 0] = i + 1
            depth = depth + instance_depth

        # import matplotlib.pyplot as plt
        # plt.imshow(mask)
        # plt.show()
        sample[constants.KEY_LABEL_DEPTHMAP] = depth[None]
        sample[constants.KEY_LABEL_INSTANCES_MASK] = mask[None]

        if self.use_cylinder:
            label_boxes_2d = sample[constants.KEY_LABEL_BOXES_2D]
            label_boxes_2d = label_boxes_2d.reshape(-1, 2)
            stereo_calib_p2 = sample[constants.KEY_STEREO_CALIB_P2]
            label_boxes_2d = image_utils.plane_to_cylinder(
                label_boxes_2d, stereo_calib_p2, self.radius).reshape(-1, 4)
            image_input = sample[constants.KEY_IMAGE]
            image_input = self.preprocess_image(image_input, stereo_calib_p2)

            # assign back
            sample[constants.KEY_IMAGE] = image_input
            sample[constants.KEY_LABEL_BOXES_2D] = label_boxes_2d
        # boxes_3d = sample[constants.KEY_LABEL_BOXES_3D]
        # boxes_2d_proj = sample[constants.KEY_LABEL_BOXES_2D]
        # p2 = sample[constants.KEY_STEREO_CALIB_P2]

        # cls_orients = []
        # reg_orients = []
        # dims = np.stack(
        # [boxes_3d[:, 4], boxes_3d[:, 5], boxes_3d[:, 3]], axis=-1)

        # for i in range(boxes_3d.shape[0]):
        # target = {}
        # target['ry'] = boxes_3d[i, -1]

        # target['dimension'] = dims[i]
        # target['location'] = boxes_3d[i, :3]

        # corners_xy, points_3d = compute_box_3d(target, p2, True)

        # # some labels for estimating orientation
        # left_side_points_2d = corners_xy[[0, 3]]
        # right_side_points_2d = corners_xy[[1, 2]]
        # left_side_points_3d = points_3d.T[[0, 3]]
        # right_side_points_3d = points_3d.T[[1, 2]]

        # # which one is visible
        # mid_left_points_3d = left_side_points_3d.mean(axis=0)
        # mid_right_points_3d = right_side_points_3d.mean(axis=0)
        # # K*T
        # KT = p2[:, -1]
        # K = p2[:3, :3]
        # T = np.dot(np.linalg.inv(K), KT)
        # C = -T
        # mid_left_dist = np.linalg.norm((C - mid_left_points_3d))
        # mid_right_dist = np.linalg.norm((C - mid_right_points_3d))
        # if mid_left_dist > mid_right_dist:
        # visible_side = right_side_points_2d
        # else:
        # visible_side = left_side_points_2d

        # cls_orient, reg_orient = truncate_box(boxes_2d_proj[i],
        # visible_side)
        # cls_orient = modify_cls_orient(cls_orient, left_side_points_2d,
        # right_side_points_2d)

        # cls_orients.append(cls_orient)
        # reg_orients.append(reg_orient)

        # sample['cls_orient'] = np.stack(cls_orients, axis=0).astype(np.int32)
        # sample['reg_orient'] = np.stack(reg_orients, axis=0).astype(np.float32)
        sample[constants.KEY_LABEL_CLASSES] = torch.from_numpy(
            sample[constants.KEY_LABEL_CLASSES]).long()

        return sample

    def _generate_orients(self, center_side):
        """
        Args:
            boxes_2d_proj: shape(N, 4)
            center_side: shape(N, 2, 2)
        """
        direction = center_side[:, 0] - center_side[:, 1]
        cond = (direction[:, 0] * direction[:, 1]) == 0
        cls_orients = np.zeros_like(cond, dtype=np.float32)
        cls_orients[cond] = -1
        cls_orients[~cond] = ((direction[~cond, 1] / direction[~cond, 0]) >
                              0).astype(np.float32)

        reg_orients = np.abs(direction)

        return np.concatenate(
            [cls_orients[..., np.newaxis], reg_orients], axis=-1)

    # def _get_center_side(self, corners_xy):
    # """
    # Args:
    # corners_xy: shape(N, 8, 2)
    # """
    # point0 = corners_xy[:, 0]
    # point1 = corners_xy[:, 1]
    # point2 = corners_xy[:, 2]
    # point3 = corners_xy[:, 3]
    # mid0 = (point0 + point1) / 2
    # mid1 = (point2 + point3) / 2
    # return np.stack([mid0, mid1], axis=1)
    def preprocess_image(self, image, p2):
        """
        Args:
            image: shape(CHW)
        """
        # PIL to cv2
        if isinstance(image, torch.Tensor):
            image = image.numpy().transpose((1, 2, 0))
            cylinder_image = image_utils.cylinder_project(
                image, p2, radus=self.radius)
            cylinder_image = cylinder_image.transpose((2, 0, 1))
            image = torch.from_numpy(cylinder_image)
        else:
            image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            cylinder_image = image_utils.cylinder_project(
                image, p2, radus=self.radius)
            image = Image.fromarray(
                cv2.cvtColor(cylinder_image, cv2.COLOR_BGR2RGB).astype(
                    np.uint8))
        return image

    def visuliaze_sample(self, sample):
        #  import ipdb
        #  ipdb.set_trace()
        image = sample[constants.KEY_IMAGE]
        num_instances = sample[constants.KEY_NUM_INSTANCES]
        # if image.shape[0] == 3:
        # image = image.permute(1, 2, 0)
        #  boxes = sample[constants.KEY_LABEL_BOXES_2D]

        label_boxes_3d = sample[constants.KEY_LABEL_BOXES_3D][:num_instances]
        p2 = sample[constants.KEY_STEREO_CALIB_P2]
        # boxes_3d_proj = geometry_utils.boxes_3d_to_boxes_2d(label_boxes_3d, p2)
        # import ipdb
        # ipdb.set_trace()
        cylinder_corners_2d = geometry_utils.boxes_3d_to_cylinder_corners_2d(
            label_boxes_3d, p2, radus=864)
        # boxes_2d = sample[constants.KEY_LABEL_BOXES_2D][:num_instances]
        # boxes_2d = boxes_3d_proj
        from utils.visualize import visualize_bbox
        from utils.drawer import ImageVisualizer
        image = np.asarray(image)
        # visualize_bbox(image, boxes_2d, save=True)
        image_dir = '/data/object/training/image_2'
        result_dir = './results/data'
        save_dir = 'results/images'
        calib_dir = '/data/object/training/calib'
        label_dir = None
        calib_file = None
        visualizer = ImageVisualizer(
            image_dir,
            result_dir,
            label_dir=label_dir,
            calib_dir=calib_dir,
            calib_file=calib_file,
            online=True,
            save_dir=save_dir)
        image_path = sample[constants.KEY_IMAGE_PATH]
        visualizer.render_image_corners_2d(
            image_path, image=image, corners_2d=cylinder_corners_2d, p2=p2)


if __name__ == '__main__':
    kitti_dataset_config = {
        'root_path': '/data',
        'data_path': 'object/training/image_2',
        'label_path': 'object/training/label_2',
        'classes': ['Car'],
        'dataset_file': './data/demo.txt',
        'use_cylinder': True,
        'radus': 864
    }
    nuscenes_dataset_config = {
        'root_path': '/data/nuscenes_kitti',
        'data_path': 'object/training/image_2',
        'label_path': 'object/training/label_2',
        'classes': ['car'],
        'dataset_file': './data/nuscenes_demo.txt'
    }
    transforms_config = [{
        "type": "random_hsv"
    }, {
        "type": "random_brightness"
    }, {
        "type": "random_horizontal_flip"
    }, {
        "size": [384, 1280],
        "type": "fix_shape_resize"
    }]
    from data import transforms
    transform = transforms.build(transforms_config)
    #  transform = None
    dataset = Mono3DKITTIDataset(
        kitti_dataset_config, transform, training=True)
    for sample in dataset:
        # sample = dataset[4]
        dataset.visuliaze_sample(sample)
    #  print(sample.keys())
