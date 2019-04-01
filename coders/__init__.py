# -*- coding: utf-8 -*-

from core.utils.imports import import_dir
import os

from core.utils.common import build_class as _build
from utils.registry import TARGET_ASSIGNERS


def build(config):
    return _build(config, TARGET_ASSIGNERS)


# import all for register all modules into registry dict
import_dir(os.path.dirname(__file__), include=['coders'])

# only export build function to outside
__all__ = ['build']
