import os
import cv2
import numpy
import torch

from PIL import Image
from data.det_dataset import DetDataset

from utils.kitti_util import *

color_map = [(0, 0, 142)]
OBJ_CLASSES = ['Car']


class KittiDataset(DetDataset):
    def __init__(self,
                 dataset_config,
                 transforms=None):
        super(KittiDataset, self).__init__()
        self.root_path = dataset_config['root_path']
        self.labels = self.make_label_list(dataset_config['dataset_file'])
        self.imgs = self.make_image_list()
        self.transforms = transforms

    def __getitem__(self, index):
        img_file = self.imgs[index]
        lbl_file = self.labels[index]

        # load image and label
        img = Image.open(img_file)
        bbox, lbl = self.read_annotation(lbl_file)

        if self.transforms is not None:
            img, bbox, lbl, im_scale = self.transforms(img, bbox, lbl)

        w = img.size()[2]
        h = img.size()[1]
        bbox[:, 2] *= w
        bbox[:, 0] *= w
        bbox[:, 1] *= h
        bbox[:, 3] *= h

        # ratio = float(w)/float(h)
        # For car, the label is one
        bbox = torch.cat((bbox, torch.ones((bbox.size()[0], 1))), dim=1)
        img_info = torch.FloatTensor([h, w, im_scale])

        return img, img_info, bbox, torch.LongTensor(
            [bbox.size()[0]]), img_file

    def read_annotation(self, file_name):
        """
        read annotation from file
        :param file_name:
        :return:boxes, labels
        boxes: [[xmin, ymin, xmax, ymax], ...]
        """
        boxes = []
        labels = []
        annos = self.load_annotation(file_name)
        for obj in annos:
            obj = obj.split(' ')
            obj[-1] = obj[-1][:-1]
            obj_name = obj[0]

            if not self.is_annotation(obj_name):
                # print obj_name
                continue

            # occluded = int(float(obj[2]))
            # if occluded > 2:
            #     continue
            #
            # truncated = float(obj[1])
            # if truncated > 0.8:
            #     continue

            obj_id = self.encode_obj_name(obj_name)
            xmin = int(float(obj[4]))
            ymin = int(float(obj[5]))
            xmax = int(float(obj[6]))
            ymax = int(float(obj[7]))
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj_id)

        boxes = numpy.array(boxes, dtype=float)
        labels = numpy.array(labels, dtype=int)

        return boxes, labels

    @staticmethod
    def load_annotation(file_name):
        with open(file_name) as f:
            bbox = f.readlines()

        return bbox

    @staticmethod
    def encode_obj_name(name):
        _id = -1
        for i in range(OBJ_CLASSES.__len__()):
            if name == OBJ_CLASSES[i]:
                _id = i
                break
        if _id == -1:
            print "wrong label !"
        return _id

    @staticmethod
    def is_label_file(filename):
        return filename.endswith(".txt")

    @staticmethod
    def is_annotation(_name):
        return any(category == _name for category in OBJ_CLASSES)

    def make_label_list(self, dataset_file):
        train_list_path = os.path.join(self.root_path, dataset_file)
        with open(train_list_path, 'r') as f:
            lines = f.readlines()
            labels = [line.strip() for line in lines]
            labels = [
                os.path.join(self.root_path, 'label_2/{}.txt'.format(label))
                for label in labels
            ]
            labels = [label for label in labels if self.__check_has_car(label)]
        return labels

    def make_image_list(self):
        images = []
        for lab in self.labels:
            lab = lab.split('/')[-1]
            lab = lab[:-4]
            img_name = lab + '.png'

            read_path = os.path.join(self.root_path,
                                     'image_2/{}'.format(img_name))
            images.append(read_path)
        return images

    def __check_has_car(self, file_path):
        lines = [line.rstrip() for line in open(file_path)]
        objs = [Object3d(line) for line in lines]
        for obj in objs:
            if obj.type == 'Car':
                return True
        return False

    def __test_load_annotation(self, annos):
        is_trusted = True
        obj_count = 0
        for obj in annos:
            obj = obj.split(' ')
            obj_name = obj[0]

            # occluded = int(float(obj[2]))
            # if occluded > 2:
            #     continue
            #
            # truncated = float(obj[1])
            # if truncated > 0.8:
            #     continue

            if self.is_annotation(obj_name):
                obj_count += 1

        if obj_count < 1:
            is_trusted = False

        return is_trusted

    @staticmethod
    def visualize_bbox(img, bbox, lbl):
        img = numpy.array(img, dtype=float)
        img = numpy.around(img)
        img = numpy.clip(img, a_min=0, a_max=255)
        img = img.astype(numpy.uint8)
        for i, box in enumerate(bbox):
            img = cv2.rectangle(
                img, (int(box[0] * 1024), int(box[1] * 512)),
                (int(box[2] * 1024), int(box[3] * 512)),
                color=color_map[lbl[i]],
                thickness=2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("test", img)
        cv2.waitKey(0)
