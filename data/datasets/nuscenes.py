# -*- coding: utf-8 -*-

import os
import json
import cv2
import numpy as np

from PIL import Image
from data.det_dataset import DetDataset
from core import constants
from utils.registry import DATASETS
from utils import geometry_utils
from utils import box_ops

NUSCENES_MEAN_DIMS = {
    'car': [3.88311640418, 1.62856739989, 1.52563191462],
    'bus': [3.88311640418, 1.62856739989, 1.52563191462],
    'truck': [10.13586957, 2.58549199, 3.2520595],
    'pedestrian': [0.84422524, 0.66068622, 1.76255119],
    'bicycle': [1.76282397, 0.59706367, 1.73698127],
    'motorcycle': [1.76282397, 0.59706367, 1.73698127],
    'trailer': [10.13586957, 2.58549199, 3.2520595],
    'construction_vehicle': [10.13586957, 2.58549199, 3.2520595],
}


@DATASETS.register('nuscenes')
class NuscenesDataset(DetDataset):
    # calib_matrix = np.asarray(
    # [[1.26641720e+03, 0.00000000e+00, 8.16267020e+02, 0.00000000e+00],
    # [0.00000000e+00, 1.26641720e+03, 4.91507066e+02, 0.00000000e+00],
    # [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]],
    # dtype=np.float32).reshape((3, 4))
    def _generate_mean_dims(self):
        mean_dims = []
        for class_type in self.classes[1:]:
            mean_dims.append(NUSCENES_MEAN_DIMS[class_type][::-1])
        return np.stack(mean_dims, axis=0).astype(np.float32)

    def __init__(self, dataset_config, transform=None, training=True):
        super().__init__(training)
        # import ipdb
        # ipdb.set_trace()
        self.transforms = transform
        self.root_path = dataset_config['root_path']
        self.data_path = os.path.join(self.root_path,
                                      dataset_config['data_path'])
        self.label_path = os.path.join(self.root_path,
                                       dataset_config['label_path'])

        self.classes = ['bg'] + dataset_config['classes']

        #  sample_name = 'n008-2018-05-21-11-06-59-0400__CAM_FRONT__1526915275512465.jpg'
        # sample_name = 'n015-2018-11-21-19-58-31+0800__CAM_FRONT__1542801715412460.jpg'
        if dataset_config.get('dataset_file') is not None:
            self.sample_names = self.make_label_list(
                os.path.join(self.label_path, dataset_config['dataset_file']))
            #  self.sample_names = {sample_name: self.sample_names[sample_name]}
            self.imgs = sorted(self.make_image_list())
        else:
            #  pass
            self.inference(
                image_dir=dataset_config.get('img_dir'),
                image_file=dataset_config.get('demo_file'))
        #  elif dataset_config['img_dir'] is not None:
        #  self.logger.info('use custom dataset')
        #  self.image_dir = dataset_config['img_dir']
        #  self.logger.info('use image dir: {}'.format(self.image_dir))
        #  self.imgs = self.load_sample_names_from_image_dir(
        #  self.image_dir)
        #  self.sample_names = self.imgs
        #  else:
        #  self.logger.info('use demo file')
        #  self.sample_names = [dataset_config['demo_file']]
        #  self.sample_names = self.imgs

        if dataset_config.get('calib_file'):
            self._calib_file = dataset_config['calib_file']
        else:
            self._calib_file = None
        self.calib_dir = self.root_path + '/calibs'

        self.max_num_boxes = 100

        # self.calif_file = dataset_config.get('calib_file')

    def _check_class(self, label):
        return label in self.classes

    def load_projection_matrix(self, calib_file):
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

    def _check_anno(self, anno):
        cats = anno['category']
        use = False
        for cat in cats:
            if self._check_class(cat):
                use = True
        return use

    def make_label_list(self, dataset_file):
        annotations = self.load_annotation(dataset_file)
        new_annotations = {}
        for anno_name, anno in annotations.items():
            if self._check_anno(anno):
                new_annotations[anno_name] = anno

        return new_annotations

    def make_image_list(self):
        imgs = []
        for anno_name, anno in self.sample_names.items():
            imgs.append(os.path.join(self.data_path, anno['name']))
        return imgs

    @staticmethod
    def load_annotation(file_name):
        with open(file_name) as f:
            anno = json.load(f)
        return anno

    def encode_obj_name(self, obj_name):
        return self.classes.index(obj_name)

    def read_annotation(self, anno):
        bboxes = []
        labels = []
        for ind, label in enumerate(anno['category']):
            if self._check_class(label):
                labels.append(self.encode_obj_name(label))
                bboxes.append(anno['box_3d'][ind])

        labels = np.asarray(labels, dtype=np.int)
        bboxes = np.asarray(bboxes, dtype=np.float32)
        return bboxes, labels

    def pad_sample(self, sample):
        label_boxes_2d = sample[constants.KEY_LABEL_BOXES_2D]
        label_boxes_3d = sample[constants.KEY_LABEL_BOXES_3D]
        label_classes = sample[constants.KEY_LABEL_CLASSES]
        all_label_boxes_3d = np.zeros((self.max_num_boxes,
                                       label_boxes_3d.shape[1]))
        all_label_boxes_2d = np.zeros((self.max_num_boxes,
                                       label_boxes_2d.shape[1]))
        all_label_classes = np.zeros((self.max_num_boxes, ))
        # assign it with bg label
        all_label_classes[...] = 0
        num_boxes = label_boxes_2d.shape[0]
        all_label_classes[:num_boxes] = label_classes
        all_label_boxes_2d[:num_boxes] = label_boxes_2d
        all_label_boxes_3d[:num_boxes] = label_boxes_3d

        sample[constants.KEY_NUM_INSTANCES] = np.asarray(
            num_boxes, dtype=np.int32)

        sample[constants.KEY_LABEL_BOXES_2D] = all_label_boxes_2d.astype(
            np.float32)
        sample[constants.KEY_LABEL_BOXES_3D] = all_label_boxes_3d.astype(
            np.float32)
        sample[constants.KEY_LABEL_CLASSES] = all_label_classes

        return sample

    def get_calib_path(self, sample_name):
        if self._calib_file is not None:
            return self._calib_file
        else:
            return os.path.join(self.calib_dir, '{}.txt'.format(sample_name))

    def get_training_sample(self, index):
        image_path = self.imgs[index]
        sample_name = os.path.basename(image_path)
        label_boxes_3d, label_classes = self.read_annotation(
            self.sample_names[sample_name])
        image = Image.open(image_path).convert('RGB')
        w, h = image.size
        image_info = np.asarray([h, w, 1.0, 1.0])

        calib_path = self.get_calib_path(sample_name[:-4])
        stereo_calib_p2 = self.load_projection_matrix(calib_path)

        ry = label_boxes_3d[:, :1]
        dim = label_boxes_3d[:, 1:4]
        location = label_boxes_3d[:, 4:7]
        label_boxes_3d = np.concatenate([location, dim, ry], axis=-1)

        # use boxes_3d_proj rather than boxes 2d
        boxes_3d_proj = geometry_utils.boxes_3d_to_boxes_2d(
            label_boxes_3d, stereo_calib_p2)
        #  boxes_2d = box_ops.np_clip_boxes(boxes_3d_proj, image_info)
        sample = {}
        sample[constants.KEY_LABEL_BOXES_2D] = boxes_3d_proj
        sample[constants.KEY_IMAGE] = image
        sample[constants.KEY_LABEL_BOXES_3D] = label_boxes_3d.astype(
            np.float32)
        sample[constants.KEY_STEREO_CALIB_P2] = stereo_calib_p2.astype(
            np.float32)
        sample[constants.KEY_LABEL_CLASSES] = label_classes.astype(np.int32)
        sample[constants.KEY_IMAGE_PATH] = image_path
        sample[constants.KEY_IMAGE_INFO] = image_info.astype(np.float32)

        mean_dims = self._generate_mean_dims()
        sample[constants.KEY_MEAN_DIMS] = mean_dims
        return sample

    def get_testing_sample(self, index):
        #  import ipdb
        #  ipdb.set_trace()
        image_path = self.imgs[index]
        image = Image.open(image_path).convert('RGB')
        w, h = image.size
        image_info = np.asarray([h, w, 1.0, 1.0])

        #  import ipdb
        #  ipdb.set_trace()
        sample_name = os.path.basename(image_path)
        calib_path = self.get_calib_path(sample_name[:-4])
        stereo_calib_p2 = self.load_projection_matrix(calib_path)
        sample = {}
        sample[constants.KEY_STEREO_CALIB_P2] = stereo_calib_p2.astype(
            np.float32)
        sample[constants.KEY_STEREO_CALIB_P2_ORIG] = np.copy(
            sample[constants.KEY_STEREO_CALIB_P2])
        sample[constants.KEY_IMAGE] = image
        sample[constants.KEY_IMAGE_PATH] = image_path
        sample[constants.KEY_IMAGE_INFO] = image_info.astype(np.float32)

        mean_dims = self._generate_mean_dims()
        sample[constants.KEY_MEAN_DIMS] = mean_dims
        return sample

    def visualize_sample(self, sample):
        image = sample[constants.KEY_IMAGE]
        bbox = sample[constants.KEY_LABEL_BOXES_2D]
        num_instances = sample[constants.KEY_NUM_INSTANCES]
        bbox = bbox[:num_instances]
        # if image.shape[0] == 3:
        # image = image.permute(1, 2, 0)
        #  boxes = sample[constants.KEY_LABEL_BOXES_2D]

        label_boxes_3d = sample[constants.KEY_LABEL_BOXES_3D]
        label_boxes_3d = label_boxes_3d[:num_instances]
        p2 = sample[constants.KEY_STEREO_CALIB_P2]
        boxes_3d_proj = geometry_utils.boxes_3d_to_boxes_2d(label_boxes_3d, p2)
        # sample[constants.KEY_LABEL_BOXES_2D] = boxes_3d_proj
        from utils.visualize import visualize_bbox
        image = np.asarray(image)
        visualize_bbox(image, boxes_3d_proj, display=True)

    @staticmethod
    def visualize_bbox(sample):
        from utils.box_vis import draw_boxes
        img = sample[constants.KEY_IMAGE]
        bbox = sample[constants.KEY_LABEL_BOXES_3D]
        num_instances = sample[constants.KEY_NUM_INSTANCES]
        p2 = sample[constants.KEY_STEREO_CALIB_P2]
        bbox = bbox[:num_instances]
        image_path = sample[constants.KEY_IMAGE_PATH]
        #  import ipdb
        #  ipdb.set_trace()

        # rearange order of cols of bbox
        location = bbox[:, :3]
        dim = bbox[:, 3:6]
        ry = bbox[:, 6:]
        bbox = np.concatenate([ry, dim, location], axis=-1)

        draw_boxes(
            img,
            bbox,
            p2,
            save_path=os.path.basename(image_path),
            box_3d_gt=None)
        # img = np.array(img, dtype=float)
        # img = np.around(img)
        # img = np.clip(img, a_min=0, a_max=255)
        # img = img.astype(np.uint8)
        # for i, box in enumerate(bbox):

    # img = cv2.rectangle(
    # img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
    # color=(55, 255, 155),
    # thickness=2)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow("test", img)
    # cv2.waitKey(0)


if __name__ == '__main__':
    import sys
    dataset_config = {
        'root_path': '/data/nuscenes',
        'dataset_file': 'trainval.json',
        'data_path': 'samples/CAM_FRONT',
        'label_path': '.',
        'classes': ['car', 'pedestrian', 'truck']
    }
    dataset = NuscenesDataset(dataset_config, training=True)
    for ind, sample in enumerate(dataset):
        dataset.visualize_bbox(sample)
        #  dataset.visualize_sample(sample)
        sys.stdout.write('\r{}/{}'.format(ind, 1000))
        sys.stdout.flush()
