# -*- coding: utf-8 -*-

import os
import json
import cv2
import numpy as np

from PIL import Image
from data.det_dataset import DetDataset
from core import constants
from utils.registry import DATASETS


@DATASETS.register('bdd')
class BDDDataset(DetDataset):
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
        self.imgs = self.make_image_list()

        self.max_num_boxes = 100

    def _check_class(self, label):
        return label in self.classes

    def _check_anno(self, anno):
        labels = anno['labels']
        use = False
        for label in labels:
            if self._check_class(label['category']):
                use = True
        return use

    def make_label_list(self, dataset_file):
        annotations = self.load_annotation(dataset_file)
        new_annotations = []
        for anno in annotations:
            if self._check_anno(anno):
                new_annotations.append(anno)

        return new_annotations[:1]

    def make_image_list(self):
        imgs = []
        for anno in self.sample_names:
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

        for label in anno['labels']:
            if label == 1:
                continue
            category = label['category']
            if self._check_class(category):
                labels.append(self.encode_obj_name(category))
                box2d = label['box2d']
                keys = ['x1', 'y1', 'x2', 'y2']
                bboxes.append([box2d[key] for key in keys])

        labels = np.asarray(labels, dtype=np.int)
        bboxes = np.asarray(bboxes, dtype=np.float32)
        return bboxes, labels

    def get_sample(self, index):
        image_path = self.imgs[index]
        label_boxes_2d, label_classes = self.read_annotation(
            self.sample_names[index])
        image = Image.open(image_path).convert('RGB')
        w, h = image.size
        image_info = np.asarray([h, w, 1.0, 1.0])

        # pad instances
        all_label_boxes_2d = np.zeros(
            (self.max_num_boxes, label_boxes_2d.shape[1]))
        all_label_classes = np.zeros((self.max_num_boxes, ))
        num_boxes = label_boxes_2d.shape[0]
        all_label_classes[:num_boxes] = label_classes
        all_label_boxes_2d[:num_boxes] = label_boxes_2d

        sample = {}
        sample[constants.KEY_IMAGE] = image
        sample[constants.KEY_LABEL_BOXES_2D] = all_label_boxes_2d.astype(
            np.float32)
        sample[constants.KEY_LABEL_CLASSES] = all_label_classes.astype(
            np.int32)
        sample[constants.KEY_IMAGE_PATH] = image_path
        sample[constants.KEY_IMAGE_INFO] = image_info.astype(np.float32)
        sample[constants.KEY_NUM_INSTANCES] = np.asarray(
            num_boxes, dtype=np.int32)
        return sample

    @staticmethod
    def visualize_bbox(img, bbox, lbl):
        img = np.array(img, dtype=float)
        img = np.around(img)
        img = np.clip(img, a_min=0, a_max=255)
        img = img.astype(np.uint8)
        for i, box in enumerate(bbox):
            img = cv2.rectangle(
                img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                color=(55, 255, 155),
                thickness=2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("test", img)
        cv2.waitKey(0)


if __name__ == '__main__':
    dataset_config = {
        'root_path': '/data/bdd/bdd100k/',
        'dataset_file': 'bdd100k_labels_images_val.json',
        'data_path': 'images/100k/val',
        'label_path': 'labels',
        'classes': ['car', 'person', 'bus']
    }
    dataset = BDDDataset(dataset_config, training=False)
    for sample in dataset:
        img = sample[constants.KEY_IMAGE]
        bbox = sample[constants.KEY_LABEL_BOXES_2D]
        num_instances = sample[constants.KEY_NUM_INSTANCES]
        bbox = bbox[:num_instances]
        dataset.visualize_bbox(img, bbox, None)
