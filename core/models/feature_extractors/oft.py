# -*- coding: utf-8 -*-
"""
use resnet18 to construct the topdown network and front end
"""

import torch.nn as nn
import torch

from torchvision import models
from core.model import Model


class OFTNetFeatureExtractor(Model):
    def init_weights(self):
        pass

    def init_param(self, model_config):
        self.model_path = model_config['pretrained_model']
        self.dout_base_model = 1024
        self.pretrained = model_config['pretrained']
        self.class_agnostic = model_config['class_agnostic']
        self.classes = model_config['classes']
        self.img_channels = model_config['img_channels']

        self.use_cascade = model_config.get('use_cascade')
        self.separate_feat = model_config.get('separate_feat')

    def init_modules(self):
        resnet = models.resnet18()
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
            resnet.layer1, resnet.layer2
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

        # img feauture
        self.img_feature = nn.Sequential(*base_features)

        # bev feature
        topdown_features = [resnet.layer3, resnet.layer4]
        self.bev_feature = nn.Sequential(*topdown_features)
