# -*- coding: utf-8 -*-

import torch.nn as nn

from core.model import Model
import copy
import os
from models import registry
from models.backbones import build_backbone


@registry.FEATURE_EXTRACTORS.register('resnet')
class ResNetFeatureExtractor(Model):
    def init_weights(self):
        pass

    def init_param(self, model_config):
        self.pretrained = model_config['pretrained']
        self.img_channels = model_config['img_channels']
        self.net_arch = model_config['net_arch']
        self.use_cascade = model_config['use_cascade']
        self.model_dir = model_config['pretrained_models_dir']
        self.model_path = os.path.join(self.model_dir,
                                       self.net_arch_path_map[self.net_arch])

    def init_modules(self):
        resnet = build_backbone(self.net_arch)()

        base_features = [
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3
        ]

        if self.separate_feat:
            base_features = base_features[:-1]
            self.first_stage_cls_feature = resnet.layer3
            self.first_stage_bbox_feature = copy.deepcopy(resnet.layer3)

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