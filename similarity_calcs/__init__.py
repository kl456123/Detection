# -*- coding: utf-8 -*-

from core.utils.imports import import_dir
from core.utils.common import build as _build
import os
from utils.registry import SIMILARITY_CALCS

include = ['center_similarity_calc']

import_dir(os.path.dirname(__file__), include=include)


def build(config):
    return _build(config, SIMILARITY_CALCS)


# only export build function to outside
__all__ = ['build']
