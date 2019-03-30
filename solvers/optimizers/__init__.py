# -*- coding: utf-8 -*-

from core.utils.imports import import_dir
from core.utils.common import build as _build
import os
from utils.registry import OPTIMIZERS

include = ['optimizers']

import_dir(os.path.dirname(__file__), include=include)


def build(config, model, logger=None):
    return _build(config, OPTIMIZERS, model, logger)


# only export build function to outside
__all__ = ['build']
