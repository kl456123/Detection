# -*- coding: utf-8 -*-

from core.utils.imports import import_dir
from core.utils.common import build as _build
import os
from utils.registry import MATCHERS

include = ['argmax_matcher', 'bipartitle_matcher']

import_dir(os.path.dirname(__file__), include=include)


def build(config):
    _build(config, MATCHERS)


# only export build function to outside
__all__ = ['build']
