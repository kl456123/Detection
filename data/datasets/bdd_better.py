# -*- coding: utf-8 -*-

import os
import json
import cv2
import numpy as np

from PIL import Image
from data.det_dataset import DetDataset
from core import constants
from utils.registry import DATASETS
from data import transforms

import torch
import anchor_generators

from utils import geometry_utils
import similarity_calcs
import bbox_coders


@DATASETS.register('bdd_better')
class BDDDataset(DetDataset):
    def __init__(self, dataset_config, transform=None, training=True):
        super().__init__(training)
        # import ipdb
        # ipdb.set_trace()
        self.transforms = transform
        self.classes = ['bg'] + dataset_config['classes']

        if dataset_config.get('img_dir') is not None:
            self.image_dir = dataset_config['img_dir']
            # directory
            self.sample_names = sorted(
                self.load_sample_names_from_image_dir(self.image_dir))
            self.imgs = self.sample_names
        elif dataset_config.get('demo_file') is not None:
            # file
            self.sample_names = sorted([dataset_config['demo_file']])
            self.imgs = self.sample_names
        else:
            # val dataset
            self.root_path = dataset_config['root_path']
            self.data_path = os.path.join(self.root_path,
                                          dataset_config['data_path'])
            self.label_path = os.path.join(self.root_path,
                                           dataset_config['label_path'])

            self.sample_names = self.make_label_list(
                os.path.join(self.label_path, dataset_config['dataset_file']))
            self.imgs = self.make_image_list()

        self.max_num_boxes = 100
        # self.default_boxes = RetinaPriorBox()(dataset_config['anchor_config'])
        self.anchor_generator = anchor_generators.build(
            dataset_config['anchor_generator_config'])
        default_boxes = self.anchor_generator.generate(
            dataset_config['input_shape'], normalize=True)

        self.default_boxes = geometry_utils.torch_xyxy_to_xywh(
            default_boxes)[0]

    @staticmethod
    def iou(box1, box2):
        '''Compute the intersection over union of two set of boxes.
        The default box order is (xmin, ymin, xmax, ymax).
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
          order: (str) box order, either 'xyxy' or 'xywh'.
        Return:
          (tensor) iou, sized [N,M].
        Reference:
          https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
        '''

        lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        iou = inter / (area1[:, None] + area2 - inter)

        return iou

    def encode_orig(self, boxes, classes, threshold=0.5):
        default_boxes = self.default_boxes
        # wh = default_boxes[:, 2:]
        default_boxes = torch.cat([
            default_boxes[:, :2] - default_boxes[:, 2:] / 2,
            default_boxes[:, :2] + default_boxes[:, 2:] / 2
        ], 1)  # xmin, ymin, xmax, ymax

        # iou = self.iou(boxes, default_boxes)  # [#obj,8732]
        similarity_calc = similarity_calcs.build({'type': 'center'})
        iou = similarity_calc.compare_batch(
            boxes.unsqueeze(0), default_boxes.unsqueeze(0)).squeeze(0)

        max_iou, max_anchor = iou.max(1)
        iou, max_idx = iou.max(0)  # [1,8732]
        max_idx.squeeze_(0)  # [8732,]
        iou.squeeze_(0)  # [8732,]

        boxes = boxes[max_idx]  # [8732,4]
        # variances = [0.1, 0.2]
        # xymin = (boxes[:, :2] - default_boxes[:, :2]) / (variances[0] * wh)
        # xymax = (boxes[:, 2:] - default_boxes[:, 2:]) / (variances[0] * wh)
        # loc = torch.cat([xymin, xymax], 1)  # [8732,4]
        coder = bbox_coders.build({'type': constants.KEY_BOXES_2D})
        loc = coder.encode_batch(
            default_boxes.unsqueeze(0), boxes.unsqueeze(0)).squeeze(0)

        neg = (iou < 0.4)
        ignore = (iou < threshold)
        os = torch.ones(iou.size()).long()
        os[ignore] = -1
        os[neg] = 0

        neg = (iou < 0.3)
        neg[max_anchor] = 0
        ignore[max_anchor] = 0
        conf = classes[max_idx]  # [8732,], background class = 0
        conf[ignore] = -1  # ignore[0.4, 0.5]
        conf[neg] = 0  # background

        return loc, conf, os, max_idx

    def encode(self, boxes, classes, threshold=0.5):
        '''Transform target bounding boxes and class labels to SSD boxes and classes. Match each object box
        to all the default boxes, pick the ones with the Jaccard-Index > 0.5:
            Jaccard(A,B) = AB / (A+B-AB)
        Args:
          boxes: (tensor) object bounding boxes (xmin,ymin,xmax,ymax) of a image, sized [#obj, 4].
          classes: (tensor) object class labels of a image, sized [#obj,].
          threshold: (float) Jaccard index threshold
        Returns:
          boxes: (tensor) bounding boxes, sized [#obj, 8732, 4].
          classes: (tensor) class labels, sized [8732,]
        '''
        default_boxes = self.default_boxes
        wh = default_boxes[:, 2:]
        default_boxes = torch.cat([
            default_boxes[:, :2] - default_boxes[:, 2:] / 2,
            default_boxes[:, :2] + default_boxes[:, 2:] / 2
        ], 1)  # xmin, ymin, xmax, ymax
        iou = self.iou(boxes, default_boxes)  # [#obj,8732]

        max_iou, max_anchor = iou.max(1)
        iou, max_idx = iou.max(0)  # [1,8732]
        max_idx.squeeze_(0)  # [8732,]
        iou.squeeze_(0)  # [8732,]

        boxes = boxes[max_idx]  # [8732,4]
        variances = [0.1, 0.2]
        xymin = (boxes[:, :2] - default_boxes[:, :2]) / (variances[0] * wh)
        xymax = (boxes[:, 2:] - default_boxes[:, 2:]) / (variances[0] * wh)
        loc = torch.cat([xymin, xymax], 1)  # [8732,4]

        neg = (iou < 0.4)
        ignore = (iou < threshold)
        os = torch.ones(iou.size()).long()
        os[ignore] = -1
        os[neg] = 0

        neg = (iou < 0.3)
        neg[max_anchor] = 0
        ignore[max_anchor] = 0
        conf = classes[max_idx]  # [8732,], background class = 0
        conf[ignore] = -1  # ignore[0.4, 0.5]
        conf[neg] = 0  # background

        return loc, conf, os, max_idx

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

        return new_annotations

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

    def pad_sample(self, sample):
        label_boxes_2d = sample[constants.KEY_LABEL_BOXES_2D]
        num_boxes = label_boxes_2d.shape[0]

        sample[constants.KEY_NUM_INSTANCES] = np.asarray(
            num_boxes, dtype=np.int32)
        label_classes = torch.from_numpy(sample[constants.KEY_LABEL_CLASSES])

        label_boxes_2d = torch.from_numpy(label_boxes_2d).float()
        target = self.encode_orig(label_boxes_2d, label_classes)
        new_sample = {}
        new_sample['gt_target'] = target
        new_sample[constants.KEY_IMAGE] = sample[constants.KEY_IMAGE]

        return new_sample

    def get_training_sample(self, index):
        image_path = self.imgs[index]
        label_boxes_2d, label_classes = self.read_annotation(
            self.sample_names[index])
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
        sample['default_boxes'] = self.default_boxes
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
        "input_shape": [384, 768],
        'root_path': '/data/bdd/bdd100k/',
        'dataset_file': 'bdd100k_labels_images_val.json',
        'data_path': 'images/100k/val',
        'label_path': 'labels',
        'classes':
        ["car", "person", "bus", "motor", "rider", "train", "truck"],
        "anchor_generator_config": {
            "type": "retina",
            "aspect_ratio": [[1.5, 3.5], [1.5, 3.5], [1.5, 3.5], [1.5, 3.5],
                             [1.5, 3.5], [1.5, 3.5]],
            "default_ratio": [0.02, 0.04, 0.08, 0.16, 0.32],
            "output_scale": [8, 16, 32, 64, 128]
        }
    }
    transform_config = [{
        "type": "color_jitter"
    }, {
        "type": "random_sample_crop_v2"
    }, {
        "type": "random_horizontal_flip_v2"
    }, {
        "type": "random_gray"
    }, {
        "type": "fix_shape_resize",
        "size": [384, 768]
    }, {
        "type": "to_tensor"
    }, {
        "type": "normalize",
        "normal_mean": [0.485, 0.456, 0.406],
        "normal_std": [0.229, 0.224, 0.225]
    }]
    transform = transforms.build(transform_config)
    dataset = BDDDataset(dataset_config, transform=transform, training=True)
    for sample in dataset:
        img = sample[constants.KEY_IMAGE]
        bbox = sample[constants.KEY_LABEL_BOXES_2D]
        num_instances = sample[constants.KEY_NUM_INSTANCES]
        bbox = bbox[:num_instances]
        dataset.visualize_bbox(img, bbox, None)
