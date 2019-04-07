# -*- coding: utf-8 -*-

from data.datasets.kitti import KITTIDataset
from core import constants
from utils import geometry_utils
import numpy as np
from utils.registry import DATASETS

KITTI_MEAN_DIMS = {
    'Car': [3.88311640418, 1.62856739989, 1.52563191462],
    'Van': [5.06763659, 1.9007158, 2.20532825],
    'Truck': [10.13586957, 2.58549199, 3.2520595],
    'Pedestrian': [0.84422524, 0.66068622, 1.76255119],
    'Person_sitting': [0.80057803, 0.5983815, 1.27450867],
    'Cyclist': [1.76282397, 0.59706367, 1.73698127],
    'Tram': [16.17150617, 2.53246914, 3.53079012],
    'Misc': [3.64300781, 1.54298177, 1.92320313]
}


@DATASETS.register('mono_3d_kitti')
class Mono3DKITTIDataset(KITTIDataset):
    def _generate_mean_dims(self):
        mean_dims = []
        for class_type in self.classes[1:]:
            mean_dims.append(KITTI_MEAN_DIMS[class_type])
        return np.stack(mean_dims, axis=0).astype(np.float32)

    def get_sample(self, index):
        sample = super().get_sample(index)
        mean_dims = self._generate_mean_dims()
        sample[constants.KEY_MEAN_DIMS] = mean_dims
        return sample
        # shape(M, 7)
        # label_boxes_3d = sample[constants.KEY_LABEL_BOXES_3D]
        # num_instances = sample[constants.KEY_NUM_INSTANCES]
        # p2 = sample[constants.KEY_STEREO_CALIB_P2]
        # label_corners_2d = geometry_utils.boxes_3d_to_corners_2d(

    # label_boxes_3d, p2)

    # shape(N, 2, 2)
    # center_side = self._get_center_side(label_corners_2d)

    # label_boxes_2d_proj = geometry_utils.corners_2d_to_boxes_2d(
    # label_corners_2d)

    # label_orients = self._generate_orients(center_side)
    # sample[constants.KEY_LABEL_ORIENTS] = label_orients
    # return sample

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
        cls_orients[~cond] = (
            (direction[~cond, 1] / direction[~cond, 0]) > 0).astype(np.float32)

        reg_orients = np.abs(direction)

        return np.concatenate(
            [cls_orients[..., np.newaxis], reg_orients], axis=-1)

    def _get_center_side(self, corners_xy):
        """
        Args:
            corners_xy: shape(N, 8, 2)
        """
        point0 = corners_xy[:, 0]
        point1 = corners_xy[:, 1]
        point2 = corners_xy[:, 2]
        point3 = corners_xy[:, 3]
        mid0 = (point0 + point1) / 2
        mid1 = (point2 + point3) / 2
        return np.stack([mid0, mid1], axis=1)


if __name__ == '__main__':
    dataset_config = {
        'root_path': '/data',
        'data_path': 'object/training/image_2',
        'label_path': 'object/training/label_2',
        'classes': ['Car', 'Pedestrian', 'Truck'],
        'dataset_file': './data/train.txt'
    }
    dataset = Mono3DKITTIDataset(dataset_config)
    sample = dataset[0]
    print(sample.keys())
