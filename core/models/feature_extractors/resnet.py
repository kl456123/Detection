# -*- coding: utf-8 -*-

import torch.nn as nn
import torch

from torchvision import models
from core.model import Model
import copy
import os


class ResNetFeatureExtractor(Model):
    def init_weights(self):
        pass

    def init_param(self, model_config):
        #  self.model_path = 'data/pretrained_model/resnet50-19c8e357.pth'
        self.model_path = os.path.join(model_config['pretrained_path'], 'resnet50-19c8e357.pth')
        self.dout_base_model = 1024
        self.pretrained = model_config['pretrained']
        self.img_channels = model_config['img_channels']

        self.use_cascade = model_config.get('use_cascade')
        # self.model_path = 'data/pretrained_model/resnet50-19c8e357.pth'
        self.separate_feat = model_config.get('separate_feat')

    def init_modules(self):
        resnet = models.resnet50()
        # self.model_path = '/node01/jobs/io/pretrained/resnet50-19c8e357.pth'
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
