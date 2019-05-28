# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.nn.functional as F

from lib.model.roi_layers import AdaptiveROIAlign

from core.model import Model
from core.filler import Filler
from core import constants

from models.losses import common_loss
from models.losses.focal_loss import FocalLoss

from utils.registry import DETECTORS
from utils import box_ops

from target_generators.target_generator import TargetGenerator
from models import feature_extractors
from models import detectors

from utils import batch_ops
import bbox_coders
from core.utils.analyzer import Analyzer


@DETECTORS.register('fpn')
class FPNFasterRCNN(Model):
    def calculate_roi_level(self, rois_batch):
        h = rois_batch[:, 4] - rois_batch[:, 2] + 1
        w = rois_batch[:, 3] - rois_batch[:, 1] + 1
        roi_level = torch.log(torch.sqrt(w * h) / 224.0)

        isnan = torch.isnan(roi_level).any()
        assert not isnan, 'incorrect value in w: {}, h: {}'.format(w, h)

        roi_level = torch.round(roi_level + 4)
        roi_level[roi_level < 2] = 2
        roi_level[roi_level > 5] = 5
        # roi_level[...] = 2
        return roi_level

    def calculate_stride_level(self, idx):
        return 1 / ((idx + 1) * 8)

    def pyramid_rcnn_pooling(self, rcnn_feat_maps, rois_batch, input_size):
        pooled_feats = []
        box_to_levels = []
        # determine which layer to get feat
        # import ipdb
        # ipdb.set_trace()
        roi_level = self.calculate_roi_level(rois_batch)
        for idx, rcnn_feat_map in enumerate(rcnn_feat_maps):
            idx += 2
            mask = roi_level == idx
            rois_batch_per_stage = rois_batch[mask]
            if rois_batch_per_stage.shape[0] == 0:
                continue
            box_to_levels.append(mask.nonzero())
            feat_map_shape = rcnn_feat_map.shape[-2:]
            stride = feat_map_shape[0] / input_size[0]
            pooled_feats.append(
                self.rcnn_pooling(rcnn_feat_map, rois_batch_per_stage, stride))

        # (Important!)Note that you should keep it original order
        pooled_feat = torch.cat(pooled_feats, dim=0)
        box_to_levels = torch.cat(box_to_levels, dim=0).squeeze()
        idx_sorted, order = torch.sort(box_to_levels)
        pooled_feat = pooled_feat[order]
        assert pooled_feat.shape[0] == rois_batch.shape[0]
        return pooled_feat

    def forward(self, feed_dict):
        # import ipdb
        # ipdb.set_trace()
        im_info = feed_dict[constants.KEY_IMAGE_INFO]

        prediction_dict = {}

        # base model
        rpn_feat_maps, rcnn_feat_maps = self.feature_extractor.first_stage_feature(
            feed_dict[constants.KEY_IMAGE])
        feed_dict.update({'rpn_feat_maps': rpn_feat_maps})

        # rpn model
        prediction_dict.update(self.rpn_model.forward(feed_dict))
        proposals = prediction_dict['proposals']
        multi_stage_loss_units = []
        # multi_stage_stats = []
        for i in range(self.num_stages):

            if self.training:
                proposals_dict = {}
                proposals_dict[constants.KEY_PRIMARY] = proposals

                # gt_dict
                gt_dict = {}
                gt_dict[constants.KEY_PRIMARY] = feed_dict[
                    constants.KEY_LABEL_BOXES_2D]
                gt_dict[constants.KEY_CLASSES] = None
                gt_dict[constants.KEY_BOXES_2D] = None

                # auxiliary_dict(used for encoding)
                auxiliary_dict = {}
                auxiliary_dict[constants.KEY_BOXES_2D] = feed_dict[
                    constants.KEY_LABEL_BOXES_2D]
                auxiliary_dict[constants.KEY_CLASSES] = feed_dict[
                    constants.KEY_LABEL_CLASSES]
                auxiliary_dict[constants.KEY_NUM_INSTANCES] = feed_dict[
                    constants.KEY_NUM_INSTANCES]
                auxiliary_dict[constants.KEY_PROPOSALS] = proposals

                proposals_dict, loss_units, stats = self.target_generators[
                    i].generate_targets(proposals_dict, gt_dict,
                                        auxiliary_dict)

                # note here base_feat (N,C,H,W),rois_batch (N,num_proposals,5)
                proposals = proposals_dict[constants.KEY_PRIMARY]
            rois = box_ops.box2rois(proposals)
            pooled_feat = self.pyramid_rcnn_pooling(rcnn_feat_maps,
                                                    rois.view(-1, 5),
                                                    im_info[0][:2])

            # shape(N,C,1,1)
            pooled_feat = self.feature_extractor.second_stage_feature(
                pooled_feat)
            pooled_feat = pooled_feat.mean(3).mean(2)

            rcnn_bbox_preds = self.rcnn_bbox_preds[i](pooled_feat)
            rcnn_cls_scores = self.rcnn_cls_preds[i](pooled_feat)

            rcnn_cls_probs = F.softmax(rcnn_cls_scores, dim=1)

            batch_size = rois.shape[0]
            rcnn_cls_scores = rcnn_cls_scores.view(batch_size, -1,
                                                   self.n_classes)
            rcnn_cls_probs = rcnn_cls_probs.view(batch_size, -1,
                                                 self.n_classes)
            if not self.class_agnostic:
                # import ipdb
                # ipdb.set_trace()
                if self.training:
                    rcnn_bbox_preds = self.squeeze_bbox_preds(
                        rcnn_bbox_preds,
                        loss_units[constants.KEY_CLASSES]['target'].view(-1))
                else:
                    rcnn_bbox_preds = self.squeeze_bbox_preds(
                        rcnn_bbox_preds,
                        rcnn_cls_probs.argmax(dim=-1).view(-1))

            rcnn_bbox_preds = rcnn_bbox_preds.view(batch_size, -1, 4)

            # decode for next stage
            coder = bbox_coders.build(self.target_generators[i]
                                      .target_generator_config['coder_config'])
            proposals = coder.decode_batch(rcnn_bbox_preds, proposals).detach()
            if self.training:
                loss_units[constants.KEY_CLASSES]['pred'] = rcnn_cls_scores
                loss_units[constants.KEY_BOXES_2D]['pred'] = rcnn_bbox_preds
                # import ipdb
                # ipdb.set_trace()
                multi_stage_loss_units.append([
                    loss_units[constants.KEY_CLASSES],
                    loss_units[constants.KEY_BOXES_2D]
                ])
                # multi_stage_stats.append(stats)

                # import ipdb
                # ipdb.set_trace()
                proposals_dict[constants.KEY_PRIMARY] = proposals
                auxiliary_dict[constants.KEY_PROPOSALS] = proposals
                _, _, second_stage_stats = self.target_generators[
                    i].generate_targets(
                        proposals_dict,
                        gt_dict,
                        auxiliary_dict,
                        subsample=False)
                fg_probs, _ = rcnn_cls_probs[:, :, 1:].max(dim=-1)
                fake_match = auxiliary_dict[constants.KEY_FAKE_MATCH]
                second_stage_stats.update(
                    Analyzer.analyze_precision(
                        fake_match,
                        fg_probs,
                        feed_dict[constants.KEY_NUM_INSTANCES],
                        thresh=0.3))

        if self.training:
            prediction_dict[constants.KEY_TARGETS] = multi_stage_loss_units
            # prediction_dict[constants.KEY_STATS] = multi_stage_stats
            prediction_dict[constants.KEY_STATS] = [stats, second_stage_stats]
        else:
            prediction_dict[constants.KEY_CLASSES] = rcnn_cls_probs

            image_info = feed_dict[constants.KEY_IMAGE_INFO]
            proposals[:, :, ::2] = proposals[:, :, ::
                                             2] / image_info[:, 3].unsqueeze(
                                                 -1).unsqueeze(-1)
            proposals[:, :, 1::2] = proposals[:, :, 1::
                                              2] / image_info[:, 2].unsqueeze(
                                                  -1).unsqueeze(-1)
            prediction_dict[constants.KEY_BOXES_2D] = proposals

        if self.training:
            loss_dict = self.loss(prediction_dict, feed_dict)
            return prediction_dict, loss_dict
        else:
            return prediction_dict

    def init_weights(self):
        # submodule init weights
        self.feature_extractor.init_weights()
        self.rpn_model.init_weights()

        for stage_ind in range(self.num_stages):
            Filler.normal_init(self.rcnn_cls_preds[stage_ind], 0, 0.01,
                               self.truncated)
            Filler.normal_init(self.rcnn_bbox_preds[stage_ind], 0, 0.001,
                               self.truncated)

    def init_modules(self):
        self.feature_extractor = feature_extractors.build(
            self.feature_extractor_config)
        self.rpn_model = detectors.build(self.rpn_config)
        self.rcnn_cls_preds = nn.ModuleList(
            [nn.Linear(1024, self.n_classes) for _ in range(self.num_stages)])
        in_channels = 1024
        if self.class_agnostic:
            rcnn_bbox_pred = nn.Linear(in_channels, 4)
        else:
            rcnn_bbox_pred = nn.Linear(in_channels, 4 * self.n_classes)

        self.rcnn_bbox_preds = nn.ModuleList(
            [rcnn_bbox_pred for _ in range(self.num_stages)])

        # loss module
        if self.use_focal_loss:
            self.rcnn_cls_loss = FocalLoss(self.n_classes, gamma=2)
        else:
            self.rcnn_cls_loss = nn.CrossEntropyLoss(reduction='none')

        self.rcnn_bbox_loss = nn.modules.SmoothL1Loss(reduction='none')

        # TODO add feat scale adaptive roi pooling
        self.rcnn_pooling = AdaptiveROIAlign(
            (self.pooling_size, self.pooling_size), 2)

    def init_param(self, model_config):
        classes = model_config['classes']
        self.classes = classes
        self.n_classes = len(classes) + 1
        self.class_agnostic = model_config['class_agnostic']
        self.pooling_size = model_config['pooling_size']
        self.pooling_mode = model_config['pooling_mode']
        self.truncated = model_config['truncated']
        self.use_focal_loss = model_config['use_focal_loss']

        # some submodule config
        self.feature_extractor_config = model_config[
            'feature_extractor_config']
        self.rpn_config = model_config['rpn_config']

        self.num_stages = len(model_config['target_generator_config'])
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

        # import ipdb
        # ipdb.set_trace()
        # submodule loss
        loss_dict.update(self.rpn_model.loss(prediction_dict, feed_dict))

        targets = prediction_dict[constants.KEY_TARGETS]

        rcnn_cls_loss = 0
        rcnn_reg_loss = 0

        for stage_ind in range(self.num_stages):
            cls_target = targets[stage_ind][0]
            # cls_targets = cls_target['target']
            # pos = cls_targets > 0  # [N,#anchors]
            # num_pos = pos.long().sum().clamp(min=1).float()

            rcnn_cls_loss = rcnn_cls_loss + common_loss.calc_loss(
                self.rcnn_cls_loss, cls_target)

            reg_target = targets[stage_ind][1]
            rcnn_reg_loss = rcnn_reg_loss + common_loss.calc_loss(
                self.rcnn_bbox_loss, reg_target)

        loss_dict.update({
            'rcnn_cls_loss': rcnn_cls_loss,
            'rcnn_reg_loss': rcnn_reg_loss
        })

        return loss_dict
