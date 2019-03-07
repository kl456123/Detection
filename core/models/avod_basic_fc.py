# -*- coding: utf-8 -*-

from core.model import Model
import torch
import torch.nn as nn
import copy
from core.filler import Filler
import torch.nn.init as init


class AVODBasicFC(Model):
    def init_param(self, model_config):
        self.fusion_method = model_config['fusion_method']
        self.num_classes = model_config['num_classes']
        self.layer_size = model_config['layer_size']
        self.keep_prob = model_config['keep_prob']
        self.layers = None
        self.output_names = ['cls', 'off', 'ang']
        self.ang_out_size = model_config['ang_out_size']
        self.off_out_size = model_config['off_out_size']

        self.ndin = model_config['ndin']
        if self.fusion_method == 'concat':
            self.ndin *= 2

    def init_weights(self):
        pass

    def init_modules(self):
        # branch network
        num_layers = len(self.layer_size)

        multi_branch = []
        for i in range(num_layers):
            layers = []
            layers.append(nn.Linear(self.ndin, self.layer_size[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout2d(self.keep_prob))
            layers = nn.Sequential(*layers)
            multi_branch.append(layers)

        self.multi_branch = nn.ModuleList(multi_branch)

        # output head
        in_channels = self.layer_size[-1]
        self.cls_pred = nn.Linear(in_channels, self.num_classes)
        self.off_pred = nn.Linear(in_channels, self.off_out_size)
        self.ang_pred = nn.Linear(in_channels, self.ang_out_size)

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

    def forward(self, input_rois, input_weights):
        # branch network
        fusion_feat = self.feature_fusion(input_rois, input_weights)
        num = fusion_feat.shape[0]
        fc_fusion = fusion_feat.view(num, -1)
        preds = []
        # import ipdb
        # ipdb.set_trace()
        for idx, output_name in enumerate(self.output_names):
            fc_drop = self.multi_branch[idx].forward(fc_fusion)

            # output head
            if output_name == 'cls':
                pred = self.cls_pred(fc_drop)
            elif output_name == 'off':
                pred = self.off_pred(fc_drop)
            elif output_name == 'ang':
                pred = self.ang_pred(fc_drop)
            preds.append(pred)

        return preds
