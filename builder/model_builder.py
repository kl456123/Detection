# -*- coding: utf-8 -*-

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from core.models.faster_rcnn_model import FasterRCNN
from core.models.two_rpn_model import TwoRPNModel
from core.models.new_faster_rcnn_model import NewFasterRCNN
from core.models.distance_faster_rcnn_model import DistanceFasterRCNN
from core.models.refine_faster_rcnn_model import RefineFasterRCNN
from core.models.gate_faster_rcnn_model import GateFasterRCNN
from core.models.iou_faster_rcnn_model import IoUFasterRCNN

# class ModelBuilder(object):
# def __init__(self, model_config):
# self.model_config = model_config

# def build(self):
# pass

# choose class or function


def build(model_config, training=True):
    net_arch = model_config['net']
    if net_arch == 'vgg16':
        fasterRCNN = vgg16(model_config)
    elif net_arch == 'resnet50':
        fasterRCNN = resnet(
            model_config,
            training, )
    elif net_arch == 'faster_rcnn':
        fasterRCNN = FasterRCNN(model_config)
    elif net_arch == 'two_rpn':
        fasterRCNN = TwoRPNModel(model_config)
    elif net_arch == 'new_faster_rcnn':
        fasterRCNN = NewFasterRCNN(model_config)
    elif net_arch == 'distance_faster_rcnn':
        fasterRCNN = DistanceFasterRCNN(model_config)
    elif net_arch == 'refine_faster_rcnn':
        fasterRCNN = RefineFasterRCNN(model_config)
    elif net_arch == 'gate_faster_rcnn':
        fasterRCNN = GateFasterRCNN(model_config)
    elif net_arch == 'iou_faster_rcnn':
        fasterRCNN = IoUFasterRCNN(model_config)
    else:
        raise ValueError('net arch {} is not supported'.format(net_arch))

    if training:
        fasterRCNN.train()
    else:
        fasterRCNN.eval()
    # depercated
    # fasterRCNN.create_architecture()
    return fasterRCNN
