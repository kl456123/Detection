# -*- coding: utf-8 -*-

from utils.registry import BBOX_CODERS
from core import constants


@BBOX_CODERS.register(constants.KEY_DIMS)
class MeanCoder(object):
    @staticmethod
    def encode_batch(gt_dims, mean_dims):
        return gt_dims - mean_dims

    @staticmethod
    def decode_batch(pred_dims, mean_dims):
        return pred_dims + mean_dims
