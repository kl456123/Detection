# -*- coding: utf-8 -*-

from core.utils.imports import import_dir
from core.utils.common import build as _build
import os
from utils.registry import BBOX_CODERS

include = ['center_coder']

import_dir(os.path.dirname(__file__), include=include)


def build(config):
    _build(config, BBOX_CODERS)


# only export build function to outside
__all__ = ['build']