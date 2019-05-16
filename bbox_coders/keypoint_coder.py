# -*- coding: utf-8 -*-

import torch

from utils.registry import BBOX_CODERS
from core import constants
from utils import geometry_utils

from core.utils import format_checker


@BBOX_CODERS.register(constants.KEY_CORNER_3D_HM)
class KeypointCoder(object):
    @staticmethod
    def encode(keypoints):
        pass

    def decode():
        pass
