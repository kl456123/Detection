# -*- coding: utf-8 -*-

import torch.nn as nn


def build(config):
    loss_type = config['type']
    if loss_type == 'ce':
        return build_ce_loss(config)
    elif loss_type == 'smooth_l1':
        return build_smooth_l1_loss(config)
    elif loss_type == 'corners_2d':
        return build_corners_2d(config)
    elif loss_type == 'corners_3d':
        return build_corners_3d(config)
    else:
        raise TypeError('unknown type : {}'.format(loss_type))


def build_ce_loss(config):
    return nn.CrossEntropyLoss(reduction='none')


def build_smooth_l1_loss(config):
    return nn.SmoothL1Loss(reduction='none')


def build_corners_2d(config):
    from models.losses.corners_loss import CornersLoss
    return CornersLoss()


def build_corners_3d(config):
    from models.losses.corners_3d_loss import Corners3DLoss
    return Corners3DLoss()
