# -*- coding: utf-8 -*-

import torch.nn as nn
import logging
from core.model import Model
import os
import torch
from utils.registry import FEATURE_EXTRACTORS
from core.filler import Filler

from models.backbones.fpn import fpn50


@FEATURE_EXTRACTORS.register('fpn_better')
class FPNFeatureExtractor(Model):
    def init_param(self, model_config):
        self.pooled_size = model_config['pooling_size']
        self.pretrained = model_config['pretrained']
        self.net_arch = model_config['net_arch']
        self.pretrained_path = model_config['pretrained_path']
        self.net_arch_path_map = {'res50': 'resnet50-19c8e357.pth'}
        self.model_path = os.path.join(self.pretrained_path,
                                       self.net_arch_path_map[self.net_arch])
        self.truncated = model_config.get('truncated', False)
        self.logger = logging.getLogger(__name__)

    def init_modules(self):
        self.fpn = fpn50()

        if self.training and self.pretrained:
            self.logger.info(
                ("Loading pretrained weights from %s" % (self.model_path)))
            state_dict = torch.load(self.model_path)
            module_dict = self.fpn.state_dict()

            checkpoint_dict = {
                k: v
                for k, v in state_dict.items() if k in module_dict
            }
            module_dict.update(checkpoint_dict)
            self.fpn.load_state_dict(module_dict)

        self.second_stage_feature = nn.Sequential(* [
            nn.Conv2d(
                256,
                1024,
                kernel_size=self.pooled_size,
                stride=self.pooled_size,
                padding=0), nn.ReLU(True), nn.Conv2d(
                    1024, 1024, kernel_size=1, stride=1, padding=0), nn.ReLU(
                        True)
        ])

    def first_stage_feature(self, inputs):
        # bottom up
        p2, p3, p4, p5 = self.fpn(inputs)

        rpn_feature_maps = [p2, p3, p4, p5]
        mrcnn_feature_maps = [p2, p3, p4, p5]
        return rpn_feature_maps, mrcnn_feature_maps
