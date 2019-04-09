# -*- coding: utf-8 -*-

from utils.registry import BBOX_CODERS
from core import constants


@BBOX_CODERS.register(constants.KEY_ORIENTS_V2)
class OrientsV2Coder(object):
    @staticmethod
    def encode_batch(label_boxes_3d, proposals, p2):
        pass

    @staticmethod
    def decode_batch():
        pass
