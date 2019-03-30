# -*- coding: utf-8 -*-

import os
from core.utils.imports import import_dir
from utils.registry import TARGET_ASSIGNERS
from core.utils.common import build as _build

include = ['target_assigner']
import_dir(os.path.dirname(__file__), include=include)


def build(config):
    return _build(config, TARGET_ASSIGNERS)


# only export build function to outside
__all__ = ['build']
