# -*- coding: utf-8 -*-

import os
from PIL import Image
import numpy as np
from data.det_dataset import DetDataset

# wavedata for kitti
from wavedata.tools.core import calib_utils
from wavedata.tools.obj_detection import obj_utils

from core import constants
from utils.registry import DATASETS
import cv2


@DATASETS.register('kitti')
class KITTIDataset(DetDataset):
    def __init__(self, config, transform=None, training=True):
        # root path of dataset
        self._root_path = os.path.join(config['root_path'], 'object/training')
        self._dataset_file = config['dataset_file']
        self._cam_idx = 2

        # set up dirs
        self._set_up_directories()

        self.transforms = transform

        # classes to be trained
        # 0 refers to bg
        classes = ['bg']
        self.classes = classes + config['classes']

        sample_names = self.load_sample_names()
        self.sample_names = sorted(self.filter_sample_names(sample_names))

        self.max_num_boxes = 40

    def _check_class(self, obj, classes):
        """This filters an object by class.
        Args:
        obj: An instance of ground-truth Object Label
        Returns: True or False depending on whether the object
        matches the desired class.
        """
        return obj.type in classes

    def load_sample_names(self):
        # set_file = './train.txt'
        with open(self._dataset_file) as f:
            sample_names = f.read().splitlines()
        return np.array(sample_names)

    def _check_difficulty(self, obj, difficulty):
        """This filters an object by difficulty.
        Args:
        obj: An instance of ground-truth Object Label
        difficulty: An int defining the KITTI difficulty rate
        Returns: True or False depending on whether the object
        matches the difficulty criteria.
        """

        return ((obj.occlusion <= self.OCCLUSION[difficulty]) and
                (obj.truncation <= self.TRUNCATION[difficulty]) and
                (obj.y2 - obj.y1) >= self.HEIGHT[difficulty])

    def filter_sample_names(self, sample_names):
        loaded_sample_names = []
        for sample_name in sample_names:
            obj_labels = obj_utils.read_labels(self.label_dir,
                                               int(sample_name))
            obj_labels = self.filter_labels(obj_labels, self.classes)
            if len(obj_labels):
                loaded_sample_names.append(sample_name)

        return loaded_sample_names

    def get_rgb_image_path(self, sample_idx):
        return os.path.join(self.image_dir, '{}.png'.format(sample_idx))

    def get_depth_map_path(self, sample_idx):
        return os.path.join(self.depth_dir, '{}.png'.format(sample_idx))

    def get_velodyne_path(self, sample_idx):
        return os.path.join(self.velo_dir, '{}.bin'.format(sample_idx))

    def filter_labels(self,
                      objects,
                      classes,
                      difficulty=None,
                      max_occlusion=None):
        """Filters ground truth labels based on class, difficulty, and
        maximum occlusion

        Args:
        objects: A list of ground truth instances of Object Label
        classes: (optional) classes to filter by, if None
        all classes are used
        difficulty: (optional) KITTI difficulty rating as integer
        max_occlusion: (optional) maximum occlusion to filter objects

        Returns:
        filtered object label list
        """

        objects = np.asanyarray(objects)
        filter_mask = np.ones(len(objects), dtype=np.bool)

        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]

            if filter_mask[obj_idx]:
                if not self._check_class(obj, classes):
                    filter_mask[obj_idx] = False
                    continue

            # Filter by difficulty (occlusion, truncation, and height)
            if difficulty is not None and \
                    not self._check_difficulty(obj, difficulty):
                filter_mask[obj_idx] = False

                continue

            if max_occlusion and \
                    obj.occlusion > max_occlusion:
                filter_mask[obj_idx] = False
                continue

        return objects[filter_mask]

    def _class_str_to_index(self, class_type):
        return self.classes.index(class_type)

    def _obj_label_to_box_3d(self, obj_label):
        """
        box_3d format: ()
        """
        box_3d = np.zeros(7)
        box_3d[3:6] = [obj_label.l, obj_label.h, obj_label.w]
        box_3d[:3] = obj_label.t
        box_3d[6] = obj_label.ry
        return box_3d

    def _obj_label_to_box_2d(self, obj_label):
        box_2d = np.zeros(4)
        box_2d = [obj_label.x1, obj_label.y1, obj_label.x2, obj_label.y2]
        return box_2d

    def get_sample(self, index):
        sample_name = self.sample_names[index]

        # image
        image_path = self.get_rgb_image_path(sample_name)
        # image_input = Image.open(image_path)
        cv_bgr_image = cv2.imread(image_path)
        image_input = cv_bgr_image[..., ::-1]
        image_shape = image_input.shape[0:2]
        # no scale now
        image_scale = (1.0, 1.0)
        image_info = image_shape + image_scale

        # calib
        stereo_calib_p2 = calib_utils.read_calibration(self.calib_dir,
                                                       int(sample_name)).p2

        # labels
        obj_labels = obj_utils.read_labels(self.label_dir, int(sample_name))
        # filter it already
        obj_labels = self.filter_labels(obj_labels, self.classes)
        label_boxes_3d = np.asarray(
            [self._obj_label_to_box_3d(obj_label) for obj_label in obj_labels])
        label_boxes_2d = np.asarray(
            [self._obj_label_to_box_2d(obj_label) for obj_label in obj_labels])
        label_classes = [
            self._class_str_to_index(obj_label.type)
            for obj_label in obj_labels
        ]
        label_classes = np.asarray(label_classes, dtype=np.int32)

        all_label_boxes_3d = np.zeros(
            (self.max_num_boxes, label_boxes_3d.shape[1]))
        all_label_boxes_2d = np.zeros(
            (self.max_num_boxes, label_boxes_2d.shape[1]))
        all_label_classes = np.zeros((self.max_num_boxes, ))
        # assign it with bg label
        all_label_classes[...] = 0
        num_boxes = label_boxes_2d.shape[0]
        all_label_classes[:num_boxes] = label_classes
        all_label_boxes_2d[:num_boxes] = label_boxes_2d
        all_label_boxes_3d[:num_boxes] = label_boxes_3d

        # image_info = list(image_info).append(num_boxes)

        transform_sample = {}
        transform_sample[constants.KEY_IMAGE] = image_input
        transform_sample[
            constants.KEY_STEREO_CALIB_P2] = stereo_calib_p2.astype(np.float32)
        transform_sample[constants.
                         KEY_LABEL_BOXES_3D] = all_label_boxes_3d.astype(
                             np.float32)
        transform_sample[constants.
                         KEY_LABEL_BOXES_2D] = all_label_boxes_2d.astype(
                             np.float32)
        transform_sample[constants.KEY_LABEL_CLASSES] = all_label_classes
        transform_sample[constants.KEY_IMAGE_PATH] = image_path

        # (h,w,scale)
        transform_sample[constants.KEY_IMAGE_INFO] = np.asarray(
            image_info, dtype=np.float32)

        transform_sample[constants.KEY_NUM_INSTANCES] = np.asarray(
            num_boxes, dtype=np.int32)
        #  import ipdb
        #  ipdb.set_trace()

        return transform_sample

    def _set_up_directories(self):
        self.image_dir = self._root_path + '/image_' + str(self._cam_idx)
        self.calib_dir = self._root_path + '/calib'
        self.disp_dir = self._root_path + 'disparity'
        self.planes_dir = self._root_path + '/planes'
        self.velo_dir = self._root_path + '/velodyne'
        self.depth_dir = self._root_path + '/depth_' + str(self._cam_idx)

        self.label_dir = self._root_path + '/label_' + str(self._cam_idx)


if __name__ == '__main__':
    dataset_config = {
        'root_path': '/data',
        'data_path': 'object/training/image_2',
        'label_path': 'object/training/label_2',
        'classes': ['Car', 'Pedestrian', 'Truck'],
        'dataset_file': './data/train.txt'
    }
    dataset = KITTIDataset(dataset_config)
    sample = dataset[0]
    print(sample.keys())
