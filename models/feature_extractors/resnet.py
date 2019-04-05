# -*- coding: utf-8 -*-

import torch.nn as nn

from core.model import Model
import copy
import os
from utils.registry import FEATURE_EXTRACTORS
from models.backbones import build_backbone
from models import feature_extractors
import torch


@FEATURE_EXTRACTORS.register('resnet')
class ResNetFeatureExtractor(Model):
    def init_weights(self):
        pass

    def init_param(self, model_config):
        self.pretrained = model_config['pretrained']
        self.img_channels = model_config['img_channels']
        self.net_arch = model_config['net_arch']
        self.use_cascade = model_config['use_cascade']
        self.pretrained_path = model_config['pretrained_path']
        self.net_arch_path_map = {'res50': 'resnet50-19c8e357.pth'}
        self.model_path = os.path.join(self.pretrained_path,
                                       self.net_arch_path_map[self.net_arch])

    def init_modules(self):
        resnet = build_backbone(self.net_arch)()
        if self.training and self.pretrained:
            print(("Loading pretrained weights from %s" % (self.model_path)))
            state_dict = torch.load(self.model_path)
            resnet.load_state_dict({
                k: v
                for k, v in list(state_dict.items())
                if k in resnet.state_dict()
            })

        base_features = [
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3
        ]

        # if not image(e.g lidar)
        if not self.img_channels == 3:
            self.first_layer = nn.Conv2d(
                self.img_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            base_features[0] = self.first_layer

        self.first_stage_feature = nn.Sequential(*base_features)

        self.second_stage_feature = nn.Sequential(resnet.layer4)
        if self.use_cascade:
            self.third_stage_feature = copy.deepcopy(self.second_stage_feature)
