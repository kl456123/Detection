# -*- coding: utf-8 -*-

from core.model import Model
import torch


class AVODFusionFC(Model):
    def init_param(self, model_config):
        self.fusion_type = model_config['fusion_type']
        self.fusion_method = model_config['fusion_method']

    def feature_fusion(self, input_rois, input_weights):
        if self.fusion_method == 'concat':
            return torch.cat(input_rois, dim=1)
        elif self.fusion_method == 'max':
            return torch.max(input_rois[0], input_rois[1])
        elif self.fusion_method == 'mean':
            rois_sum = torch.sum(input_rois)
            weights = input_weights[0] + input_weights[1]
            return rois_sum / weights
        else:
            raise ValueError(
                'unknown fusion method {}'.format(self.fusion_method))

    def init_weights(self):
        pass

    def init_modules(self):
        pass

    def forward(self, input_rois, input_weights):
        fusion_feat = self.feature_fusion(input_rois, input_weights)
        # flatten
        num = fusion_feat.shape[0]
        fc_fusion = fusion_feat.view(num, -1)
