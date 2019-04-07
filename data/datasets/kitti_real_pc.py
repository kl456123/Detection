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
from core.anchor_generators.grid_anchor_3d_generator import GridAnchor3dGenerator
from utils import kitti_aug
from core.avod import box_3d_encoder
from core.avod import anchor_filter
from core.avod import anchor_projector

from utils import pc_ops


class PointCloudKittiDataset(DetDataset):
    # OBJ_CLASSES = ['Car']

    def __init__(self, dataset_config, transforms=None, training=True):
        super(PointCloudKittiDataset, self).__init__(training)
        self._root_path = dataset_config['root_path']
        self._dataset_file = dataset_config['dataset_file']
        self._area_extents = dataset_config['area_extents']
        self._bev_extents = [self._area_extents[0], self._area_extents[2]]
        self._voxel_size = 0.1
        self.transforms = transforms

        self._cam_idx = 2
        self._set_up_directories()

        # self.bev_generator = BevGenerator(
        # dataset_config['bev_generator_config'])
        self.bev_generator = BevSlices(dataset_config['bev_generator_config'])

        self.classes = ['bg'] + dataset_config['classes']

        self.loaded_sample_names = self.load_sample_names()

        # filter sample names first
        self.filter_sample_names()

        self.imgs = self.loaded_sample_names

        self.use_pc = dataset_config.get('use_pc')

        self.anchor_generator = GridAnchor3dGenerator(
            dataset_config['anchor_generator_config'])

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

        return self.classes.index(class_type)

    def load_sample_names(self):
        set_file = os.path.join(self._root_path, self._dataset_file)
        #  set_file = './train.txt'
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
        label_boxes_3d = np.asarray([
            box_3d_encoder.object_label_to_box_3d(obj_label)
            for obj_label in obj_labels
        ])
        label_classes = [
            self._class_str_to_index(obj_label.type)
            for obj_label in obj_labels
        ]
        label_classes = np.asarray(label_classes, dtype=np.int32)

        # point_cloud = self.get_point_cloud(sample_name)
        im_size = (image_shape[1], image_shape[0])
        point_cloud = obj_utils.get_lidar_point_cloud(
            int(sample_name), self.calib_dir, self.velo_dir, im_size=im_size)

        # bev maps
        bev_images = self.bev_generator.generate_bev(point_cloud, ground_plane,
                                                     self._area_extents)

        height_maps = bev_images.get('height_maps')
        density_map = bev_images.get('density_map')
        bev_input = np.dstack((*height_maps, density_map))

        bev_input = bev_input.transpose(2, 0, 1)
        bev_shape = bev_input.shape[-2:]

        # Generate anchors
        grid_anchor_boxes_3d = self.anchor_generator.generate(ground_plane)
        voxel_grid_2d = pc_ops.create_sliced_voxel_grid_2d(
            point_cloud, self._area_extents, self._voxel_size, ground_plane)

        all_anchor_boxes_3d = grid_anchor_boxes_3d

        anchors_to_use = box_3d_encoder.box_3d_to_anchor(all_anchor_boxes_3d)
        empty_filter = anchor_filter.get_empty_anchor_filter_2d(
            anchors_to_use, voxel_grid_2d, density_threshold=1)
        anchor_boxes_3d_to_use = all_anchor_boxes_3d[empty_filter]

        anchor_boxes_3d_to_use = np.asarray(anchor_boxes_3d_to_use)

        anchors_to_use = box_3d_encoder.box_3d_to_anchor(
            anchor_boxes_3d_to_use)
        # num_anchors = len(anchors_to_use)

        # Project anchors to bev and img
        bev_anchors, bev_anchors_norm = anchor_projector.project_to_bev(
            anchors_to_use, self._bev_extents)
        img_anchors, img_anchors_norm = anchor_projector.project_to_image_space(
            anchors_to_use, stereo_calib_p2, image_shape)

        # generate anchor iou and offset
        label_anchors = box_3d_encoder.box_3d_to_anchor(
            label_boxes_3d, ortho_rotate=True)
        img_anchors_gt, img_anchors_gt_norm = anchor_projector.project_to_image_space(
            label_anchors, stereo_calib_p2, image_shape)
        bev_anchors_gt, bev_anchors_gt_norm = anchor_projector.project_to_bev(
            label_anchors, self._bev_extents)

        transform_sample = {}
        transform_sample['bev_input'] = bev_input.astype(np.float32)
        transform_sample['img'] = image_input.astype(np.float32)
        transform_sample['stereo_calib_p2'] = stereo_calib_p2.astype(
            np.float32)
        transform_sample['ground_plane'] = ground_plane.astype(np.float32)
        transform_sample['point_cloud'] = point_cloud.astype(np.float32)
        transform_sample['label_boxes_3d'] = label_boxes_3d.astype(np.float32)
        transform_sample['label_classes'] = label_classes
        transform_sample['label_anchors'] = label_anchors.astype(np.float32)
        transform_sample['img_name'] = img_path
        transform_sample['img_orig'] = image_input.astype(np.float32)
        transform_sample['im_info'] = [1, 1, 1]
        # transform_sample['image_shape'] = image_shape
        transform_sample['img_anchors_gt_norm'] = img_anchors_gt_norm.astype(
            np.float32)

        # anchors info
        # transform_sample['bev_anchors'] = bev_anchors.astype(np.float32)
        # transform_sample['img_anchors'] = img_anchors.astype(np.float32)
        transform_sample['bev_anchors_norm'] = bev_anchors_norm.astype(
            np.float32)
        transform_sample['img_anchors_norm'] = img_anchors_norm.astype(
            np.float32)
        transform_sample['bev_anchors_gt_norm'] = bev_anchors_gt_norm.astype(
            np.float32)
        # transform_sample['bev_anchors_gt'] = bev_anchors_gt.astype(np.float32)
        transform_sample['anchors'] = anchors_to_use.astype(np.float32)

        transform_sample['origin_image_shape'] = np.asarray(
            image_shape).astype(np.float32)
        # transform_sample['bev_shape'] = np.asarray(bev_shape).astype(
        # np.float32)

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
        "anchor_generator_config": {
            "area_extents": [[-40, 40], [-5, 3], [0, 70.4]],
            "anchor_offset": [0, 0],
            "anchor_stride": [0.5, 0.5],
            "anchor_size": [[3.4, 1.7, 1.5]]
        },
        "cache_bev": False,
        "dataset_file": "train.txt",
        "root_path": "/data/object/training",
        "classes": ["Car"],
        "bev_generator_config": {
            "height_lo": -0.2,
            "height_hi": 2.3,
            "num_slices": 5,
            "voxel_size": 0.1
        },
        "area_extents": [[-40, 40], [-5, 3], [0, 70.4]]
    }
    dataset = PointCloudKittiDataset(dataset_config)
    # import ipdb
    # ipdb.set_trace()
    # sample = dataset[0]
    import sys
    num = len(dataset)
    for i in range(num):
        sample = dataset[i]
        if sample['img_anchors_norm'].shape[0] == 0:
            import ipdb
            ipdb.set_trace()
        sys.stdout.write('\r{}/{}'.format(i + 1, num))
        sys.stdout.flush()
