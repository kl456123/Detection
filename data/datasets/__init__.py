# -*- coding: utf-8 -*-

from core.utils.imports import import_dir
from core.utils.common import build as _build
import os
from utils.registry import DATASETS

include = [
    'coco', 'kitti', 'mono_3d_kitti', 'bdd', 'bdd_better', 'nuscenes',
    'inference_dataset', 'keypoint_kitti', 'cityscape'
]

import_dir(os.path.dirname(__file__), include=include)


def build(config, transform, training):

    return _build(config, DATASETS, transform, training)


# only export build function to outside
__all__ = ['build']
