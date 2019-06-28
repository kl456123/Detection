# -*- coding: utf-8 -*-

from core.utils.imports import import_dir
import os

from core.utils.common import build as _build
from utils.registry import DETECTORS


def build(config):
    return _build(config, DETECTORS)


include = [
    'faster_rcnn_model', 'rpn_model', 'mono_3d_model', 'fpn_faster_rcnn_model',
    'fpn_rpn_model', 'fpn_mono_3d_model', 'fpn_multibin_model',
    'fpn_mono_3d_rear_model', 'pr_model', 'pr_mono_3d_model', 'fpn_corners_model',
    'fpn_grnet', 'fpn_mono_3d_better_model', 'rpn_model_test', 'fpn_corners_stable_model',
    'maskrcnn_model', 'fpn_corners_3d_model', 'fpn_corner_loss', 'fpn_grnet_reverse',
    'fpn_rpn_grnet_model', 'mobileye_model'
]
# import all for register all modules into registry dict
import_dir(os.path.dirname(__file__), include=include)

# only export build function to outside
__all__ = ['build']
