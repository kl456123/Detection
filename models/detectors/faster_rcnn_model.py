# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F

# from lib.model.roi_align.modules.roi_align import RoIAlignAvg

from core.model import Model
from core.filler import Filler
from core import constants

from models.losses import common_loss
from models.losses.focal_loss import FocalLoss
from lib.model.roi_align.modules.roi_align import RoIAlignAvg

from utils.registry import DETECTORS
from utils import box_ops

from target_generators.target_generator import TargetGenerator
from models import feature_extractors
from models import detectors


@DETECTORS.register('faster_rcnn')
class FasterRCNN(Model):
    def forward(self, feed_dict):

        prediction_dict = {}

        # base model
        base_feat = self.feature_extractor.first_stage_feature(
            feed_dict[constants.KEY_IMAGE])
        feed_dict.update({'base_feat': base_feat})
        self.add_feat('base_feat', base_feat)

        # rpn model
        prediction_dict.update(self.rpn_model.forward(feed_dict))
        proposals = prediction_dict['proposals']
        multi_stage_targets = []
        for i in range(self.num_stages):
            proposals, targets = self.target_generators[i].generate_targets(
                proposals, feed_dict)

            # note here base_feat (N,C,H,W),rois_batch (N,num_proposals,5)
            rois = box_ops.box2rois(proposals)
            pooled_feat = self.rcnn_pooling(base_feat, rois.view(-1, 5))

            # shape(N,C,1,1)
            pooled_feat = self.feature_extractor.second_stage_feature(
                pooled_feat)
            pooled_feat = pooled_feat.mean(3).mean(2)

            rcnn_bbox_preds = self.rcnn_bbox_pred(pooled_feat)
            rcnn_cls_scores = self.rcnn_cls_pred(pooled_feat)

            rcnn_cls_probs = F.softmax(rcnn_cls_scores, dim=1)

            targets[0]['pred'] = rcnn_cls_probs
            targets[1]['pred'] = rcnn_bbox_preds
            multi_stage_targets.extend(targets)

        prediction_dict[constants.KEY_TARGETS] = targets

        return prediction_dict

    def init_weights(self):
        # submodule init weights
        self.feature_extractor.init_weights()
        self.rpn_model.init_weights()

        Filler.normal_init(self.rcnn_cls_pred, 0, 0.01, self.truncated)
        Filler.normal_init(self.rcnn_bbox_pred, 0, 0.001, self.truncated)

    def init_modules(self):
        self.feature_extractor = feature_extractors.build(
            self.feature_extractor_config)
        self.rpn_model = detectors.build(self.rpn_config)
        if self.pooling_mode == 'align':
            self.rcnn_pooling = RoIAlignAvg(self.pooling_size,
                                            self.pooling_size, 1.0 / 16.0)
        self.rcnn_cls_pred = nn.Linear(2048, self.n_classes)
        if self.reduce:
            in_channels = 2048
        else:
            in_channels = 2048 * 4 * 4
        if self.class_agnostic:
            self.rcnn_bbox_pred = nn.Linear(in_channels, 4)
        else:
            self.rcnn_bbox_pred = nn.Linear(in_channels, 4 * self.n_classes)

        # loss module
        if self.use_focal_loss:
            self.rcnn_cls_loss = FocalLoss(self.num_classes)
        else:
            self.rcnn_cls_loss = common_loss.CrossEntropyLoss(reduce=False)

        self.rcnn_bbox_loss = nn.modules.SmoothL1Loss(reduce=False)

    def init_param(self, model_config):
        self.num_stages = model_config['num_stages']
        classes = model_config['classes']
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = model_config['class_agnostic']
        self.pooling_size = model_config['pooling_size']
        self.pooling_mode = model_config['pooling_mode']
        self.truncated = model_config['truncated']

        self.use_focal_loss = model_config['use_focal_loss']
        self.rcnn_batch_size = model_config['batch_size']

        # some submodule config
        self.feature_extractor_config = model_config['feature_extractor_config']
        self.rpn_config = model_config['rpn_config']

        self.target_generators = [
            TargetGenerator(model_config['target_generator_config'][i])
            for i in range(self.num_stages)
        ]

    def loss(self, prediction_dict, feed_dict):
        """
        assign proposals label and subsample from them
        Then calculate loss
        """
        loss_dict = {}

        # submodule loss
        loss_dict.update(self.rpn_model.loss(prediction_dict, feed_dict))

        targets = prediction_dict[constants.KEY_TARGETS]

        cls_target = targets[0]
        reg_target = targets[1]

        rcnn_cls_loss = self.rcnn_cls_loss(*cls_target)
        rcnn_reg_loss = self.rcnn_bbox_loss(*reg_target)
        loss_dict.update({
            'rcnn_cls_loss': rcnn_cls_loss,
            'rcnn_reg_loss': rcnn_reg_loss
        })

        return loss_dict