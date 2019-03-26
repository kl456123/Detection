# -*- coding: utf-8 -*-

import os
import numpy as np

from PIL import Image

from data.det_dataset import DetDataset
from core import constants


class CocoDataset(DetDataset):
    def __init__(self, config, transform=None, training=True):
        # root path of dataset
        self._root_path = os.path.join(config['root_path'])
        self._data_path = os.path.join(self._root_path, config['data_path'])
        self._label_path = os.path.join(self._root_path, config['label_path'])

        from pycocotools.coco import COCO
        self.coco = COCO(self._label_path)

        # set up dirs
        # self._set_up_directories()

        self.transforms = transform

        # classes to be trained
        self.classes = config['classes']

        # import ipdb
        # ipdb.set_trace()
        self.class_ids = self.coco.getCatIds(catNms=self.classes)

        sample_names = self.load_sample_names()
        self.sample_names = self.filter_sample_names(sample_names)

    def _check_class(self, obj, classes):
        """This filters an object by class.
        Args:
        obj: An instance of ground-truth Object Label
        Returns: True or False depending on whether the object
        matches the desired class.
        """
        return obj in classes

    def _class_str_to_index(self, class_type):
        return self.class_ids.index(class_type)

    def load_sample_names(self):
        sample_names = list(self.coco.imgs.keys())
        return sample_names

    def read_annotation(self, targets):
        """
        read annotation from file
        :param targets:
        :return:boxes, labels
        boxes: [[xmin, ymin, xmax, ymax], ...]
        """
        boxes = []
        labels = []
        for obj in targets:
            obj_id = obj['category_id']
            if not self._check_class(obj_id, self.class_ids):
                continue
            obj_id = self._class_str_to_index(obj_id)
            box = obj['bbox']
            boxes.append([box[0], box[1], box[0] + box[2], box[1] + box[3]])
            labels.append(obj_id)

        boxes = np.array(boxes, dtype=float)
        labels = np.array(labels, dtype=int)

        return boxes, labels

    def filter_sample_names(self, sample_names):
        loaded_sample_names = []
        for sample_name in sample_names:
            anno_id = self.coco.getAnnIds(imgIds=sample_name)
            targets = self.coco.loadAnns(anno_id)
            label_boxes_2d, label_classes = self.read_annotation(targets)
            if len(label_boxes_2d) > 0:
                loaded_sample_names.append(sample_name)
        return loaded_sample_names

    def get_sample(self, index):
        coco = self.coco
        img_id = self.sample_names[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        bbox, lbl = self.read_annotation(target)

        path = coco.loadImgs(img_id)[0]['file_name']
        image_path = os.path.join(self._data_path, path)
        image = Image.open(image_path).convert('RGB')

        sample = {}
        sample[constants.KEY_IMAGE] = image
        sample[constants.KEY_LABEL_BOXES_2D] = bbox
        sample[constants.KEY_LABEL_CLASSES] = lbl
        sample[constants.KEY_IMAGE_PATH] = image_path

        return sample


if __name__ == '__main__':
    dataset_config = {
        'root_path': '/data/liangxiong/COCO2017/',
        'data_path': 'train2017',
        'label_path': 'annotations/instances_train2017.json',
        'classes': ['car']
    }
    dataset = CocoDataset(dataset_config)
    sample = dataset[0]
    print(len(dataset))
