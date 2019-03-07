import os
import cv2
import numpy as np
import torch
import random

from PIL import Image
from data.det_dataset import DetDataset
from utils.box_vis import load_projection_matrix
from utils.kitti_util import *
import cv2
from wavedata.tools.core import calib_utils
from wavedata.tools.obj_detection import obj_utils
from core.bev_generator import BevGenerator
from core.avod.bev_slices import BevSlices
from utils import kitti_aug


class PointCloudKittiDataset(DetDataset):
    OBJ_CLASSES = ['Car']

    def __init__(self, dataset_config, transforms=None, training=True):
        super(PointCloudKittiDataset, self).__init__(training)
        self._root_path = dataset_config['root_path']
        self._dataset_file = dataset_config['dataset_file']
        self._area_extents = dataset_config['area_extents']
        self.transforms = transforms

        self._cam_idx = 2
        self._set_up_directories()

        # self.bev_generator = BevGenerator(
        # dataset_config['bev_generator_config'])
        self.bev_generator = BevSlices(dataset_config['bev_generator_config'])

        self.classes = dataset_config['classes']

        self.loaded_sample_names = self.load_sample_names()

        # filter sample names first
        self.filter_sample_names()

        self.imgs = self.loaded_sample_names

        self.use_pc = dataset_config.get('use_pc')

    def _set_up_directories(self):
        self.image_dir = self._root_path + '/image_' + str(self._cam_idx)
        self.calib_dir = self._root_path + '/calib'
        self.disp_dir = self._root_path + 'disparity'
        self.planes_dir = self._root_path + '/planes'
        self.velo_dir = self._root_path + '/velodyne'
        self.depth_dir = self._root_path + '/depth_' + str(self._cam_idx)

        self._label_dir = self._root_path + '/label_' + str(self._cam_idx)

    def get_rgb_image_path(self, sample_idx):
        return os.path.join(self.image_dir, '{}.png'.format(sample_idx))

    def get_depth_map_path(self, sample_idx):
        return os.path.join(self.depth_dir, '{}.png'.format(sample_idx))

    def get_velodyne_path(self, sample_idx):
        return os.path.join(self.velo_dir, '{}.bin'.format(sample_idx))

    def filter_labels(self,
                      objects,
                      classes=None,
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
        if classes is None:
            classes = self.dataset.classes

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

    def _check_class(self, obj, classes):
        """This filters an object by class.
        Args:
        obj: An instance of ground-truth Object Label
        Returns: True or False depending on whether the object
        matches the desired class.
        """
        return obj.type in classes

    def _class_str_to_index(self, class_type):
        # if class_type == 'Car':
        # return 0

        return self.OBJ_CLASSES.index(class_type)

    def _obj_label_to_box_3d(self, obj_label):
        """
        box_3d format: ()
        """
        box_3d = np.zeros(7)
        box_3d[3:6] = [obj_label.l, obj_label.h, obj_label.w]
        box_3d[:3] = obj_label.t
        box_3d[6] = obj_label.ry
        return box_3d

    def load_sample_names(self):
        set_file = os.path.join(self._root_path, self._dataset_file)
        # set_file = './demo.txt'
        with open(set_file) as f:
            sample_names = f.read().splitlines()
        return np.array(sample_names)

    def filter_sample_names(self):
        loaded_sample_names = []
        for sample_name in self.loaded_sample_names:
            obj_labels = obj_utils.read_labels(self._label_dir,
                                               int(sample_name))
            obj_labels = self.filter_labels(obj_labels, self.classes)
            if len(obj_labels):
                loaded_sample_names.append(sample_name)
        self.loaded_sample_names = loaded_sample_names

        return loaded_sample_names

    def get_point_cloud(self, sample_name):
        """The point cloud should be projected to rect coordinates
        """
        pc_file_path = os.path.join(self.velo_dir,
                                    '{}.bin'.format(sample_name))
        points = np.fromfile(pc_file_path, dtype=np.float32).reshape((-1, 4))

        # Project points to rect coordinates
        calib_file_path = os.path.join(self.calib_dir,
                                       '{}.txt'.format(sample_name))
        calib = Calibration(calib_file_path)
        points_rect = calib.project_velo_to_rect(points[:, :3])
        points[:, :3] = points_rect

        return points_rect

    def get_transform_sample(self, index):

        sample_name = self.loaded_sample_names[index]
        # image
        img_path = self.get_rgb_image_path(sample_name)
        cv_bgr_image = cv2.imread(img_path)
        rgb_image = cv_bgr_image[..., ::-1]
        image_shape = rgb_image.shape[0:2]
        image_input = rgb_image

        # ground plane
        ground_plane = obj_utils.get_road_plane(
            int(sample_name), self.planes_dir)

        # calibration
        stereo_calib_p2 = calib_utils.read_calibration(self.calib_dir,
                                                       int(sample_name)).p2

        # labels
        obj_labels = obj_utils.read_labels(self._label_dir, int(sample_name))
        # filter it already
        obj_labels = self.filter_labels(obj_labels, self.classes)
        label_boxes_3d = np.asarray(
            [self._obj_label_to_box_3d(obj_label) for obj_label in obj_labels])
        label_classes = [
            self._class_str_to_index(obj_label.type)
            for obj_label in obj_labels
        ]
        label_classes = np.asarray(label_classes, dtype=np.int32)

        # point cloud
        # (w,h) in wavedata
        # im_size = [image_shape[1], image_shape[0]]
        # point_cloud = obj_utils.get_lidar_point_cloud(
        # int(sample_name), self.calib_dir, self.velo_dir, im_size=im_size).T
        # import ipdb
        # ipdb.set_trace()
        point_cloud = self.get_point_cloud(sample_name)

        if random.random() < 0.5:
            image_input = kitti_aug.flip_image(image_input)
            point_cloud = kitti_aug.flip_point_cloud(point_cloud)
            obj_labels = [
                kitti_aug.flip_label_in_3d_only(obj) for obj in obj_labels
            ]
            ground_plane = kitti_aug.flip_ground_plane(ground_plane)
            stereo_calib_p2 = kitti_aug.flip_stereo_calib_p2(stereo_calib_p2,
                                                             image_shape)

        if random.random() < 0.5:
            # pca jitter
            image_input[:, :, 0:3] = kitti_aug.apply_pca_jitter(
                image_input[:, :, 0:3])

        # import ipdb
        # ipdb.set_trace()
        # bev maps
        bev_images = self.bev_generator.generate_bev(
            point_cloud.transpose(), ground_plane, self._area_extents)

        # stack height maps and density map
        # height_maps = bev_images.get('height_maps')
        # density_map = bev_images.get('density_map')
        # bev_input = np.dstack((*height_maps, density_map))
        # bev_input = bev_input.transpose((2, 0, 1)).astype(np.float32)
        bev_input = bev_images.astype(np.float32)

        transform_sample = {}
        transform_sample['bev_input'] = bev_input.astype(np.float32)
        transform_sample['img'] = image_input.astype(np.float32)
        transform_sample['stereo_calib_p2'] = stereo_calib_p2.astype(
            np.float32)
        transform_sample['ground_plane'] = ground_plane.astype(np.float32)
        transform_sample['point_cloud'] = point_cloud.astype(np.float32)
        transform_sample['label_boxes_3d'] = label_boxes_3d.astype(np.float32)
        transform_sample['label_classes'] = label_classes
        transform_sample['img_name'] = img_path
        transform_sample['img_orig'] = image_input.astype(np.float32)
        transform_sample['im_info'] = [1, 1, 1]

        return transform_sample

    def __getitem__(self, index):

        transform_sample = self.get_transform_sample(index)

        if self.transforms is not None:
            transform_sample = self.transforms(transform_sample)

        return self.get_training_sample(transform_sample)

    def get_training_sample(self, transform_sample):
        return transform_sample


if __name__ == '__main__':
    dataset_config = {
        'classes': ['Car'],
        'dataset_file': './train.txt',
        'bev_generator_config': {
            'height_lo': -0.3,
            'height_hi': 3.7,
            'num_slices': 5,
            'voxel_size': 0.5
        },
        'root_path': '/data/object/training',
        'area_extents': [[-40, 40], [-5, 3], [0, 70]]
    }
    dataset = PointCloudKittiDataset(dataset_config)
    import ipdb
    ipdb.set_trace()
    sample = dataset[0]
