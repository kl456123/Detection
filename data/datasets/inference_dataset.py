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


@DATASETS.register('infer_3d')
class Infer3Dataset(DetDataset):
    def __init__(self, config, transform=None, logger=None):
        super().__init__(False)
        """
        three input args
        1. single image
        2. image dir
        3. dataset file(contain many images)
        """

        # image data source
        if config.get('img_dir') is not None:
            # image dir
            self.image_dir = config['img_dir']
            # image full path
            self.sample_names = self.load_sample_names_from_image_dir(
                self.image_dir)
        elif config.get('img_path') is not None:
            self.sample_names = [config['img_path']]
        elif config.get('img_setfile') is not None:
            pass
        else:
            raise ValueError('no any data souce is specificd !')

        # calib params
        if config.get('calib_dir') is not None:
            self.calib_dir = config['calib_dir']
        elif config.get('calib_path') is not None:
            self.calib_file = config['calib_path']
        elif config.get('calib_setfile') is not None:
            pass

    def get_calib_path(self, sample_name):
        if self.calib_file is not None:
            return self.calib_file
        else:
            return os.path.join(self.calib_dir, '{}.txt'.format(sample_name))

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

    def get_sample(self, index):
        # get image
        image_path = self.sample_names[index]
        sample_name = self.get_sample_name_from_path(image_path)
        image_input = Image.open(image_path)
        image_shape = image_input.size[::-1]
        # no scale now
        image_scale = (1.0, 1.0)
        image_info = image_shape + image_scale

        # get calib
        calib_path = self.get_calib_path(sample_name)
        stereo_calib_p2 = self.load_projection_matrix(calib_path)

        transform_sample = {}
        transform_sample[constants.KEY_IMAGE] = image_input
        transform_sample[
            constants.KEY_STEREO_CALIB_P2] = stereo_calib_p2.astype(np.float32)
        transform_sample[constants.KEY_IMAGE_PATH] = image_path

        # (h,w,scale)
        transform_sample[constants.KEY_IMAGE_INFO] = np.asarray(
            image_info, dtype=np.float32)

        return transform_sample
