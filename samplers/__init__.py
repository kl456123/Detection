# -*- coding: utf-8 -*-

from core.utils.imports import import_dir
import os

from core.utils.common import build as _build
from utils.registry import SAMPLERS


def build(config):
    return _build(config, SAMPLERS)


# import all for register all modules into registry dict
import_dir(os.path.dirname(__file__), include=['balanced_sampler'])

# only export build function to outside
__all__ = ['build']
