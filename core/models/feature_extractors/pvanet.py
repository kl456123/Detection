# -*- coding: utf-8 -*-

import torch.nn as nn
import torch

from core.model import Model
from models.pvanet import pvanet
from collections import OrderedDict


def copyStateDict(state_dict):
    if "module." in list(state_dict.keys())[
            0]:  # This is when weight file is saved with multi gpu
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        return new_state_dict
    else:
        return state_dict


class PVANetFeatureExtractor(Model):
    def init_weights(self):
        pass

    def init_param(self, model_config):
        self.pretrained = model_config['pretrained']
        self.class_agnostic = model_config['class_agnostic']
        self.classes = model_config['classes']
        self.img_channels = model_config['img_channels']

        self.use_cascade = model_config.get('use_cascade')
        self.model_path = model_config['pretrained_model']
        self.separate_feat = model_config.get('separate_feat')

    def init_modules(self):
        import ipdb
        ipdb.set_trace()
        resnet = pvanet()
        if self.training and self.pretrained:
            print(("Loading pretrained weights from %s" % (self.model_path)))
            state_dict = torch.load(self.model_path)['state_dict']
            resnet.load_state_dict(copyStateDict(state_dict))
            #  resnet.load_state_dict({
        #  k: v
        #  for k, v in list(state_dict.items())
        #  if k in resnet.state_dict()
        #  })

        feats = resnet.features
        # 1/16
        base_features = [feats.conv1, feats.conv2, feats.conv3, feats.conv4]

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

        head = [nn.Linear(12544, 4096), resnet.classifier[1:-1]]

        self.second_stage_feature = nn.Sequential(*head)
