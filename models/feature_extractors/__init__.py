# -*- coding: utf-8 -*-

from core.utils.imports import import_dir
from core.utils.common import build as _build
import os
from utils.registry import FEATURE_EXTRACTORS

exclude = ['pvanet', 'mobilenet']
import_dir(os.path.dirname(__file__), exclude=exclude)


def build(config):
    _build(config, FEATURE_EXTRACTORS)


# only export build function to outside
__all__ = ['build']
