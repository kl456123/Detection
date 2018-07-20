# -*- coding: utf-8 -*-

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

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
        fasterRCNN = resnet(model_config)
    else:
        raise ValueError('net arch {} is not supported'.format(net_arch))

    if training:
        fasterRCNN.train()
    else:
        fasterRCNN.eval()
    fasterRCNN.create_architecture()
    return fasterRCNN
