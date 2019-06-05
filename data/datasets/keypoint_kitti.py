# -*- coding: utf-8 -*-

from data.datasets.kitti import KITTIDataset
from core import constants
from utils import geometry_utils
import numpy as np
from utils.registry import DATASETS
import torch


@DATASETS.register('keypoint_kitti')
class KeyPointKittiDataset(KITTIDataset):
    def get_training_sample(self, index):
        sample = super().get_training_sample(index)
        label_boxes_3d = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_3D])
        p2 = torch.from_numpy(sample[constants.KEY_STEREO_CALIB_P2])
        image_info = torch.from_numpy(sample[constants.KEY_IMAGE_INFO])
        keypoint = self._generate_keypoint(label_boxes_3d, p2, image_info)
        sample[constants.KEY_KEYPOINTS] = keypoint
        return sample

    def get_testing_sample(self, index):
        sample = super().get_testing_sample(index)
        sample[constants.KEY_STEREO_CALIB_P2_ORIG] = np.copy(
            sample[constants.KEY_STEREO_CALIB_P2])
        return sample

    def pad_sample(self, sample):
        sample = super().pad_sample(sample)
        keypoints = sample[constants.KEY_KEYPOINTS]
        num_boxes = keypoints.shape[0]
        all_keypoints = torch.zeros(self.max_num_boxes, keypoints.shape[1],
                                    keypoints.shape[2])
        all_keypoints[:num_boxes] = keypoints
        sample[constants.KEY_KEYPOINTS] = all_keypoints.view(
            self.max_num_boxes, -1)

        return sample

    def _generate_keypoint(self, label_boxes_3d, p2, image_info):
        """
            Args:
        """
        # get keypoint

        corners_2d = geometry_utils.torch_boxes_3d_to_corners_2d(
            label_boxes_3d, p2)

        # get visibility
        image_shape = torch.tensor([0, 0, image_info[1], image_info[0]])
        image_shape = image_shape.type_as(corners_2d).view(1, 4)
        image_filter = geometry_utils.torch_window_filter(
            corners_2d, image_shape, deltas=200)

        keypoint = torch.cat(
            [corners_2d, image_filter.unsqueeze(-1).float()], dim=-1)
        return keypoint
