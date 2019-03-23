# -*- coding: utf-8 -*-

from core.model import Model
from core.models.feature_extractors.pyramid_vggnet import PyramidVggnetExtractor
from core.target_assigner import TargetAssigner
import torch.nn as nn
import torch
import torch.nn.functional as F


class SSDModel(Model):
    def init_params(self, model_config):
        self.feature_extractor_config = model_config['feature_extractor_config']
        self.multibox_cfg = model_config['multibox_cfg']

        self.target_assigner = TargetAssigner(
            model_config['target_assigner_config'])

    def init_module(self):
        self.feature_extractor = PyramidVggnetExtractor(
            self.feature_extractor_config)

        # loc layers and conf layers
        base_feat = self.feature_extractor.base_feat
        extra_layers = self.feature_extractor.extras_layers
        loc_layers, conf_layers = self.make_multibox(base_feat, extra_layers)
        self.loc_layers = loc_layers
        self.conf_layers = conf_layers

        # loss layers
        self.loc_loss = F.smooth_l1_loss()
        self.conf_loss = nn.CrossEntropyLoss(reduce=False)

    def make_multibox(self, vgg, extra_layers):
        cfg = self.multibox_cfg
        num_classes = self.n_classes
        loc_layers = []
        conf_layers = []
        vgg_source = [21, -2]
        for k, v in enumerate(vgg_source):
            loc_layers += [
                nn.Conv2d(
                    vgg[v].out_channels, cfg[k] * 4, kernel_size=3, padding=1)
            ]
            conf_layers += [
                nn.Conv2d(
                    vgg[v].out_channels,
                    cfg[k] * num_classes,
                    kernel_size=3,
                    padding=1)
            ]
        for k, v in enumerate(extra_layers[1::2], 2):
            loc_layers += [
                nn.Conv2d(
                    v.out_channels, cfg[k] * 4, kernel_size=3, padding=1)
            ]
            conf_layers += [
                nn.Conv2d(
                    v.out_channels,
                    cfg[k] * num_classes,
                    kernel_size=3,
                    padding=1)
            ]
        return loc_layers, conf_layers

    def init_weights(self):
        pass

    def forward(self, feed_dict):
        img = feed_dict['img']
        source_feats = self.feature_extractor(img)
        loc_preds = []
        conf_preds = []

        # apply multibox head to source layers
        for (x, l, c) in zip(source_feats, self.loc_layers, self.conf_layers):
            loc_preds.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf_preds.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc_preds = torch.cat([o.view(o.size(0), -1) for o in loc_preds], 1)
        conf_preds = torch.cat([o.view(o.size(0), -1) for o in conf_preds], 1)
        prediction_dict = {'loc_preds': loc_preds, 'conf_preds': conf_preds}
        return prediction_dict

    def loss(self, prediction_dict, feed_dict):
        loc_targets, conf_targets = self.target_assigner.assign()

        # ohem
        batch_sampled_mask = self.sampler.subsample()

        loc_preds = prediction_dict['loc_preds']
        # loc loss
        loc_loss = self.loc_loss(loc_preds, loc_targets)

        conf_preds = prediction_dict['conf_preds']
        # conf loss
        conf_loss = self.conf_loss(conf_preds, conf_targets)

        loss_dict = {'loc_loss': loc_loss, 'conf_loss': conf_loss}
        return loss_dict
