# -*- coding: utf-8 -*-

import logging
import torch.nn as nn
import torch

from core.model import Model
import os
from utils.registry import FEATURE_EXTRACTORS
from models.backbones import build_backbone
from models import feature_extractors
import torch.nn.functional as F
from models.backbones.squeeze_resnet_feature import SqueezeResNetFeature, BasicBlock
from torch.nn import BatchNorm2d as bn


class StridePoolBlock(nn.Sequential):
    def __init__(self, input_num, out_num, stride=2):
        super(StridePoolBlock, self).__init__()
        self.add_module(
            'conv3',
            nn.Conv2d(
                in_channels=input_num,
                out_channels=out_num,
                kernel_size=3,
                stride=stride,
                padding=1))
        self.add_module('bn', bn(num_features=out_num))
        self.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, _input):
        _feature = super(StridePoolBlock, self).forward(_input)

        return _feature


@FEATURE_EXTRACTORS.register('prnet')
class PRNetFeatureExtractor(Model):
    def init_weights(self):
        self.logger.info(
            ("Loading pretrained weights from %s" % (self.model_path)))
        state_dict = torch.load(self.model_path)
        self.features.load_state_dict(state_dict, strict=False)

    def init_param(self, model_config):
        self.pretrained = model_config['pretrained']
        self.img_channels = model_config['img_channels']
        self.net_arch = model_config['net_arch']
        self.use_cascade = model_config['use_cascade']
        self.pretrained_path = model_config['pretrained_path']
        self.net_arch_path_map = {
            'res50': 'resnet50-19c8e357.pth',
            'res18_pruned': 'resnet18_pruned0.5.pth'
        }
        self.model_path = os.path.join(self.pretrained_path,
                                       self.net_arch_path_map[self.net_arch])
        self.logger = logging.getLogger(__name__)
        self.pyramid_layers = len(model_config['output_scale'])
        self.det_features = model_config['det_features']
        # including bg
        self.num_class = len(model_config['classes']) + 1
        self.dla_input = model_config['dla_input']
        self.layer_structure = model_config['layer_structure']

    def init_modules(self):
        det_features = self.det_features
        self.features = SqueezeResNetFeature(BasicBlock, self.layer_structure,
                                             0.5, det_features, self.dla_input)
        self.det_feature1 = StridePoolBlock(
            det_features, det_features, stride=1)
        self.det_feature2 = StridePoolBlock(det_features, det_features)
        self.det_feature3 = StridePoolBlock(det_features, det_features)
        self.det_feature4 = StridePoolBlock(det_features, det_features)
        self.det_feature5 = StridePoolBlock(det_features, det_features)
        if self.pyramid_layers == 6:
            self.det_feature6 = StridePoolBlock(det_features, det_features)

    def forward(self, _input):
        feature = self.features.forward(_input)
        if self.pyramid_layers == 6:
            feature = F.interpolate(feature, scale_factor=2, mode='bilinear')

        det_feature = []
        feature = self.det_feature1(feature)
        det_feature.append(feature)

        feature = self.det_feature2(feature)
        det_feature.append(feature)

        feature = self.det_feature3(feature)
        det_feature.append(feature)

        feature = self.det_feature4(feature)
        det_feature.append(feature)

        feature = self.det_feature5(feature)
        det_feature.append(feature)

        if self.pyramid_layers == 6:
            feature = self.det_feature6(feature)
            det_feature.append(feature)

        # loc1_preds, loc2_preds, os_preds, cls_preds = self.multibox(
        # det_feature)
        return det_feature
