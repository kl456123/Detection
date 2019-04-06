# -*- coding: utf-8 -*-

from data.datasets.kitti import KITTIDataset
from core import constants
from utils import geometry_utils
import numpy as np
from utils.registry import DATASETS


@DATASETS.register('mono_3d_kitti')
class Mono3DKITTIDataset(KITTIDataset):
    def get_sample(self, index):
        sample = super().get_sample(index)
        # shape(M, 7)
        label_boxes_3d = sample[constants.KEY_LABEL_BOXES_3D]
        # num_instances = sample[constants.KEY_NUM_INSTANCES]
        p2 = sample[constants.KEY_STEREO_CALIB_P2]
        label_corners_2d = geometry_utils.boxes_3d_to_corners_2d(
            label_boxes_3d, p2)

        # shape(N, 2, 2)
        center_side = self._get_center_side(label_corners_2d)

        # label_boxes_2d_proj = geometry_utils.corners_2d_to_boxes_2d(
        # label_corners_2d)

        label_orients = self._generate_orients(center_side)
        sample[constants.KEY_LABEL_ORIENTS] = label_orients
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
