# -*- coding: utf-8 -*-

from core.utils.imports import import_dir
from core.utils.common import build_class as _build
import os
from utils.registry import BBOX_CODERS

include = [
    'center_coder', 'mean_coder', 'orient_coder', 'orientv3_coder',
    'orientv2_coder', 'rear_side_coder', 'corner_coder', 'bbox_3d_coder',
    'corner_2d_coder', 'corner_3d_coder', 'nearest_corner_coder',
    'nearestv2_corner_coder', 'mono_3d_coder', 'depth_coder', 'corner_2d_stable_coder',
    'keypoint_coder', 'hm_coder', 'gr_coder'
]

import_dir(os.path.dirname(__file__), include=include)


def build(config):
    return _build(config, BBOX_CODERS)


# only export build function to outside
__all__ = ['build']
