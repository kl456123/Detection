# -*- coding: utf-8 -*-

import os
import json
import cv2
import numpy as np

from PIL import Image
from data.det_dataset import DetDataset
from core import constants
from utils.registry import DATASETS


@DATASETS.register('nuscenes')
class NuscenesDataset(DetDataset):
    calib_matrix = np.asarray(
        [[1.26641720e+03, 0.00000000e+00, 8.16267020e+02, 0.00000000e+00],
         [0.00000000e+00, 1.26641720e+03, 4.91507066e+02, 0.00000000e+00],
         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]],
        dtype=np.float32).reshape((3, 4))

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
        self.sample_names = self.make_label_list(
            os.path.join(self.label_path, dataset_config['dataset_file']))
        self.imgs = sorted(self.make_image_list())

        self.max_num_boxes = 100

        # self.calif_file = dataset_config.get('calib_file')

    def _check_class(self, label):
        return label in self.classes

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
        label_classes = sample[constants.KEY_LABEL_CLASSES]
        all_label_boxes_2d = np.zeros(
            (self.max_num_boxes, label_boxes_2d.shape[1]))
        all_label_classes = np.zeros((self.max_num_boxes, ))
        # assign it with bg label
        all_label_classes[...] = 0
        num_boxes = label_boxes_2d.shape[0]
        all_label_classes[:num_boxes] = label_classes
        all_label_boxes_2d[:num_boxes] = label_boxes_2d

        sample[constants.KEY_NUM_INSTANCES] = np.asarray(
            num_boxes, dtype=np.int32)

        sample[constants.KEY_LABEL_BOXES_2D] = all_label_boxes_2d.astype(
            np.float32)
        sample[constants.KEY_LABEL_CLASSES] = all_label_classes
        sample[constants.KEY_STEREO_CALIB_P2] = self.calib_matrix
        return sample

    def get_training_sample(self, index):
        image_path = self.imgs[index]
        sample_name = os.path.basename(image_path)
        label_boxes_2d, label_classes = self.read_annotation(
            self.sample_names[sample_name])
        image = Image.open(image_path).convert('RGB')
        w, h = image.size
        image_info = np.asarray([h, w, 1.0, 1.0])

        sample = {}
        sample[constants.KEY_IMAGE] = image
        sample[constants.KEY_LABEL_BOXES_2D] = label_boxes_2d.astype(
            np.float32)
        sample[constants.KEY_LABEL_CLASSES] = label_classes.astype(np.int32)
        sample[constants.KEY_IMAGE_PATH] = image_path
        sample[constants.KEY_IMAGE_INFO] = image_info.astype(np.float32)
        return sample

    def get_testing_sample(self, index):
        image_path = self.imgs[index]
        image = Image.open(image_path).convert('RGB')
        w, h = image.size
        image_info = np.asarray([h, w, 1.0, 1.0])

        sample = {}
        sample[constants.KEY_IMAGE] = image
        sample[constants.KEY_IMAGE_PATH] = image_path
        sample[constants.KEY_IMAGE_INFO] = image_info.astype(np.float32)
        return sample

    @staticmethod
    def visualize_bbox(sample):
        from utils.box_vis import draw_boxes
        img = sample[constants.KEY_IMAGE]
        bbox = sample[constants.KEY_LABEL_BOXES_2D]
        num_instances = sample[constants.KEY_NUM_INSTANCES]
        p2 = sample[constants.KEY_STEREO_CALIB_P2]
        # (xyz, wlh, ry)
        bbox = bbox[:num_instances]
        image_path = sample[constants.KEY_IMAGE_PATH]

        # import ipdb
        # ipdb.set_trace()
        z = bbox[:, 0]
        x = -bbox[:, 1]
        y = -bbox[:, 2]
        l = bbox[:, 3]
        w = bbox[:, 4]
        h = bbox[:, 5]

        new_bbox = np.zeros_like(bbox)
        new_bbox[:, 0] = bbox[:, 6]
        # (hwl)
        new_bbox[:, 1:4] = np.stack([h, w, l], axis=-1)
        new_bbox[:, 4:7] = bbox[:, :3]
        # np.stack([y, z, x], axis=-1)
        draw_boxes(
            img,
            new_bbox,
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
    dataset_config = {
        'root_path': '/data/nuscenes',
        'dataset_file': 'nuscenes_3d.json',
        'data_path': 'samples/CAM_FRONT',
        'label_path': '.',
        'classes': ['car', 'pedestrian', 'truck']
    }
    dataset = NuscenesDataset(dataset_config, training=True)
    for sample in dataset:
        dataset.visualize_bbox(sample)
