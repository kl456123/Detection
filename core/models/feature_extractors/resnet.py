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
        self.dout_base_model = 1024
        self.pretrained = model_config['pretrained']
        self.class_agnostic = model_config['class_agnostic']
        self.classes = model_config['classes']
        self.img_channels = model_config['img_channels']

        self.use_cascade = model_config.get('use_cascade')
        self.separate_feat = model_config.get('separate_feat')
        self.net_arch = model_config['net_arch']
        self.model_dir = 'data/pretrained_model'
        # pretrained model in node server
        # self.model_dir = '/node01/jobs/io/pretrained'
        self.net_arch_path_map = {
            'res18': 'resnet18-5c106cde.pth',
            'res34': 'resnet34-333f7ec4.pth',
            'res50': 'resnet50-19c8e357.pth',
            'res101': 'resnet101-5d3b4d8f.pth',
            'res152': 'resnet152-b121ed2d.pth',
        }
        self.net_arch_model_map = {
            'res18': models.resnet18,
            'res34': models.resnet34,
            'res50': models.resnet50,
            'res101': models.resnet101,
            'res152': models.resnet152
        }
        self.model_path = os.path.join(self.model_dir,
                                       self.net_arch_path_map[self.net_arch])

    def init_modules(self):
        resnet = self.net_arch_model_map[self.net_arch]()

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
