# -*- coding: utf-8 -*-

from core.utils.imports import import_dir
import os

from core.utils.common import build as _build
from utils.registry import TRANSFORMS
from torchvision import transforms as trans


def build(config):
    """
    building transform is different from all above,
    it needs combine components of transforms
    """
    transforms = []
    for trans_config in config:
        transforms.append(_build(trans_config, TRANSFORMS))
    return trans.Compose(transforms)


# import all for register all modules into registry dict
import_dir(os.path.dirname(__file__))

# only export build function to outside
__all__ = ['build']
