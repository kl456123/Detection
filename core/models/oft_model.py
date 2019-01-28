# -*- coding: utf-8 -*-
"""
use one stage detector as the framework to detect 3d object
in OFT feature map
"""

import torch
from core.model import Model
from core.models.feature_extractors.oft import OFTNetFeatureExtractor
from core.models.voxel_generator import VoxelGenerator


class OFTModel(Model):
    def forward(self, feed_dict):
        img_feat_maps = self.feature_extractor.img_feature(feed_dict['img'])

    def init_param(self, model_config):
        pass

    def init_modules(self):
        """
        some modules
        """

        self.feature_extractor = OFTNetFeatureExtractor(
            self.feature_extractor_config)

        self.voxel_generator = VoxelGenerator(self.voxel_generator_config)

    def init_weights(self):
        self.feature_extractor.init_weights()

    def loss(self, prediction_dict, feed_dict):
        pass
