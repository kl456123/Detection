# -*- coding: utf-8 -*-

from core.utils.imports import import_dir
from core.utils.common import build_class as _build
import os
from utils.registry import BBOX_CODERS

include = [
    'center_coder', 'mean_coder', 'orient_coder', 'orientv3_coder',
    'orientv2_coder', 'rear_side_coder', 'corner_coder'
]

import_dir(os.path.dirname(__file__), include=include)


def build(config):
    return _build(config, BBOX_CODERS)


# only export build function to outside
__all__ = ['build']
