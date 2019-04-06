# -*- coding: utf-8 -*-

from utils.registry import BBOX_CODERS
from core import constants

KITTI_MEAN_DIMS = {
    'Car': [3.88311640418, 1.62856739989, 1.52563191462],
    'Van': [5.06763659, 1.9007158, 2.20532825],
    'Truck': [10.13586957, 2.58549199, 3.2520595],
    'Pedestrian': [0.84422524, 0.66068622, 1.76255119],
    'Person_sitting': [0.80057803, 0.5983815, 1.27450867],
    'Cyclist': [1.76282397, 0.59706367, 1.73698127],
    'Tram': [16.17150617, 2.53246914, 3.53079012],
    'Misc': [3.64300781, 1.54298177, 1.92320313]
}


@BBOX_CODERS.register(constants.KEY_DIMS)
class MeanCoder(object):
    @staticmethod
    def encode_batch(gt_dims):
        return gt_dims - KITTI_MEAN_DIMS

    @staticmethod
    def decode_batch(pred_dims):
        return pred_dims + KITTI_MEAN_DIMS
