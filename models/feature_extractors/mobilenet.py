# -*- coding: utf-8 -*-

import torch.nn as nn
from core.model import Model
import torch
from models.mobilenet_v2 import mobilenetv2


class MobileNetFeatureExtractor(Model):
    def init_weights(self):
        pass

    def init_param(self, model_config):
        #  self.model_path = 'data/pretrained_model/resnet50-19c8e357.pth'
        self.dout_base_model = 1024
        self.pretrained = model_config['pretrained']
        self.class_agnostic = model_config['class_agnostic']
        self.classes = model_config['classes']
        self.img_channels = model_config['img_channels']

        self.model_path = model_config['pretrained_model']

    def init_modules(self):
        #  import ipdb
        #  ipdb.set_trace()
        resnet = mobilenetv2()
        if self.training and self.pretrained:
            print(("Loading pretrained weights from %s" % (self.model_path)))
            state_dict = torch.load(self.model_path)
            resnet.load_state_dict({
                k: v
                for k, v in list(state_dict.items())
                if k in resnet.state_dict()
            })

        feat = resnet.features
        base_features = feat[:14]

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

        #  import ipdb
        #  ipdb.set_trace()
        head = [feat[14:16]]

        self.second_stage_feature = nn.Sequential(*head)
