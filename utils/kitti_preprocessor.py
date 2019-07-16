# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import sys

from wavedata.tools.core import calib_utils
from wavedata.tools.obj_detection import obj_utils
from utils import image_utils
from data.datasets.kitti import KITTIDataset
from core import constants


class KittiPreprocessor(object):
    def __init__(self):
        dataset_config = {
            'root_path': '/data',
            'data_path': 'object/training/image_2',
            'label_path': 'object/training/label_2',
            'classes': ['Car', 'Pedestrian', 'Truck'],
            'dataset_file': './data/train.txt'
        }
        self.saved_dir = '/data/kitti_cylinder'
        self.dataset = KITTIDataset(dataset_config)
        self.radus = 842

    def preprocess_image(self, image, p2):
        cylinder_image = image_utils.cylinder_project(
            image, p2, radus=self.radus)
        return cylinder_image

    def preprocess_label(self, label_boxes_2d, p2):
        label_boxes_2d = label_boxes_2d.reshape(-1, 2)
        cylinder_label_boxes_2d = image_utils.plane_to_cylinder(
            label_boxes_2d, p2, self.radus).reshape(-1, 4)
        return cylinder_label_boxes_2d

    def init_dirs(self):
        pass

    def _get_saved_image_path(self, sample_name):
        return os.path.join(
            self.saved_dir,
            'object/training/image_2/{}.png'.format(sample_name))

    def _get_saved_label_path(self, sample_name):
        return os.path.join(
            self.saved_dir,
            'object/training/label_2/{}.txt'.format(sample_name))

    def start(self):

        for ind, sample in enumerate(self.dataset):
            image_path = sample[constants.KEY_IMAGE_PATH]
            sample_name = os.path.basename(image_path)[:-4]
            image = cv2.imread(image_path)
            p2 = sample[constants.KEY_STEREO_CALIB_P2]
            # get image
            cylinder_image = self.preprocess_image(image, p2)
            saved_image_path = self._get_saved_image_path(sample_name)
            cv2.imwrite(saved_image_path, cylinder_image)
            sys.stdout.write('\r{}'.format(ind))
            sys.stdout.flush()

            # get label
            #  label_boxes_2d = sample[constants.KEY_LABEL_BOXES_2D]
            #  label_boxes_3d = sample[constants.KEY_LABEL_BOXES_3D]
            #  num_instances = sample[constants.KEY_NUM_INSTANCES]
            #  cylinder_label_boxes_2d = self.preprocess_label(label_boxes_2d, p2)
            #  saved_label_path = self._get_saved_label_path(sample_name)
            #  import ipdb
            #  ipdb.set_trace()
            #  cylinder_label = np.concatenate([cylinder_label_boxes_2d,
                                            #  label_boxes_3d], axis=-1)
            #  self.save_dets(cylinder_label[:num_instances], saved_label_path)

    def save_dets(self, dets, label_path):

        res_str = []
        kitti_template = '{} -1 -1 -10 {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'
        with open(label_path, 'w') as f:
            for det in dets:
                xmin, ymin, xmax, ymax, cls_ind, h, w, l, x, y, z, ry = det
                res_str.append(
                    kitti_template.format(self.dataset.classes[cls_ind], xmin,
                                          ymin, xmax, ymax, h, w, l, x, y, z,
                                          ry))
            f.write('\n'.join(res_str))


if __name__ == '__main__':
    preprocessor = KittiPreprocessor()
    preprocessor.start()
