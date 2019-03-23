# -*- coding: utf-8 -*-
"""
Utils for generating backbone
"""
from torchvision import models

_net_arch_model_map = {
    'res18': models.resnet18,
    'res34': models.resnet34,
    'res50': models.resnet50,
    'res101': models.resnet101,
    'res152': models.resnet152
}

_net_arch_fn_map = {
    'res18': 'resnet18-5c106cde.pth',
    'res34': 'resnet34-333f7ec4.pth',
    'res50': 'resnet50-19c8e357.pth',
    'res101': 'resnet101-5d3b4d8f.pth',
    'res152': 'resnet152-b121ed2d.pth',
}


def build_backbone(net_arch):
    return _net_arch_model_map[net_arch]


def build_weights_fname(net_arch):
    _net_arch_fn_map[net_arch]
