# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import Model
from core.models.avod_mono_rpn_model import RPNModel
from core.models.focal_loss import FocalLoss
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.psroi_pooling.modules.psroi_pool import PSRoIPool

from core.filler import Filler
from core.avod_target_assigner import TargetAssigner
from core.samplers.hard_negative_sampler import HardNegativeSampler
from core.samplers.balanced_sampler import BalancedSampler
from core.models.feature_extractors.resnet import ResNetFeatureExtractor
from core.samplers.detection_sampler import DetectionSampler
from core.models.avod_basic_fc import AVODBasicFC
from core.profiler import Profiler
from core.anchor_projector import AnchorProjector
from lib.model.nms.nms_wrapper import nms

import functools
from utils import box_ops


class AVODMonoFasterRCNN(Model):
    def forward(self, feed_dict):
        # import ipdb
        # ipdb.set_trace()
        img_feat_maps = self.feature_extractor.first_stage_feature(
            feed_dict['img'])
        feed_dict['img_feat_maps'] = img_feat_maps

        prediction_dict = {}

        # rpn model
        prediction_dict.update(self.rpn_model.forward(feed_dict))

        top_proposals = prediction_dict['proposals_batch']

        # expand rois
        expand_length = self.expand_proposals_xz
        if expand_length > 0:
            expanded_dim_x = top_proposals[:, :, 3] + expand_length
            expanded_dim_z = top_proposals[:, :, 5] + expand_length
            expanded_rois_batch = torch.stack(
                [
                    top_proposals[:, :, 0], top_proposals[:, :, 1],
                    top_proposals[:, :, 2], expanded_dim_x,
                    top_proposals[:, :, 4], expanded_dim_z
                ],
                dim=-1)
            avod_projection_in = expanded_rois_batch
        else:
            avod_projection_in = top_proposals

        # img
        img_proposal_boxes = self.anchor_projector.project_to_image_space(
            avod_projection_in, feed_dict['p2'])
        roi_indexes = torch.zeros_like(img_proposal_boxes[:, -1:])
        img_rois_boxes = torch.cat([roi_indexes, img_proposal_boxes], dim=-1)

        # roi crop

        # path drop
        if not (self.path_drop_probabilities[0] ==
                self.path_drop_probabilities[1] == 1.0):
            img_mask = prediction_dict['img_mask']
            img_feat_maps = img_mask * img_feat_maps
        else:
            img_mask = 1.0

        img_rois = self.rcnn_pooling(img_feat_maps, img_rois_boxes)

        # output
        pooled_feat = self.feature_extractor.second_stage_feature(img_rois)
        pooled_feat = pooled_feat.mean(3).mean(2)
        all_cls_logits = self.rcnn_cls_pred(pooled_feat)
        all_reg_3d = self.rcnn_reg_pred(pooled_feat)

        all_cls_softmax = F.softmax(all_cls_logits, dim=-1)

        prediction_dict['all_cls_logits'] = all_cls_logits
        prediction_dict['all_reg_3d'] = all_reg_3d
        prediction_dict['all_cls_softmax'] = all_cls_softmax

        # used for matching
        prediction_dict['proposals_batch'] = top_proposals

        # when inference
        if not self.training:
            # import ipdb
            # ipdb.set_trace()
            final_bboxes_3d = self.bbox_coder.decode_batch(all_reg_3d,
                                                           top_proposals)
            # nms for final detection
            final_bboxes_bev = self.anchor_projector.project_to_bev(
                final_bboxes_3d, self.area_extents)
            keep_idx, _ = box_ops.nms(final_bboxes_bev, all_cls_softmax[:, -1],
                                      self.nms_thresh)
            keep_idx = keep_idx.long().view(-1)
            final_bboxes_3d = final_bboxes_3d[keep_idx]
            all_cls_softmax = all_cls_softmax[keep_idx]
            prediction_dict['final_bboxes_3d'] = final_bboxes_3d
            prediction_dict['all_cls_softmax'] = all_cls_softmax

        return prediction_dict

    def init_weights(self):
        # submodule init weights
        self.feature_extractor.init_weights()
        self.rpn_model.init_weights()

        Filler.normal_init(self.rcnn_cls_pred, 0, 0.01, self.truncated)
        Filler.normal_init(self.rcnn_reg_pred, 0, 0.001, self.truncated)

    def init_modules(self):
        self.feature_extractor = ResNetFeatureExtractor(
            self.feature_extractor_config)
        self.rpn_model = RPNModel(self.rpn_config)
        if self.pooling_mode == 'align':
            self.rcnn_pooling = RoIAlignAvg(self.pooling_size,
                                            self.pooling_size, 1.0 / 16.0)
        elif self.pooling_mode == 'ps':
            self.rcnn_pooling = PSRoIPool(7, 7, 1.0 / 16, 7, self.n_classes)
        elif self.pooling_mode == 'psalign':
            raise NotImplementedError('have not implemented yet!')
        elif self.pooling_mode == 'deformable_psalign':
            raise NotImplementedError('have not implemented yet!')

        self.rcnn_cls_pred = nn.Linear(2048, self.n_classes)
        self.rcnn_reg_pred = nn.Linear(2048, 8)

        # loss module
        if self.use_focal_loss:
            self.rcnn_cls_loss = FocalLoss(
                self.n_classes, alpha=0.25, gamma=2, auto_alpha=False)
        else:
            self.rcnn_cls_loss = nn.CrossEntropyLoss(reduce=False)
            # self.rcnn_cls_loss = functools.partial(
        # F.cross_entropy, reduce=False)

        self.rcnn_bbox_loss = nn.modules.SmoothL1Loss(reduce=False)

        self.anchor_projector = AnchorProjector()

    def init_param(self, model_config):
        self.nms_thresh = model_config['nms_thresh']
        self.path_drop_probabilities = model_config['path_drop_probabilities']
        self.area_extents = model_config['area_extents']
        self.bev_extents = [self.area_extents[0], self.area_extents[2]]

        classes = model_config['classes']
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = model_config['class_agnostic']
        self.pooling_size = model_config['pooling_size']
        self.pooling_mode = model_config['pooling_mode']
        self.truncated = model_config['truncated']
        self.expand_proposals_xz = model_config['expand_proposals_xz']

        self.use_focal_loss = model_config['use_focal_loss']
        self.rcnn_batch_size = model_config['rcnn_batch_size']

        # some submodule config
        self.feature_extractor_config = model_config['feature_extractor_config']
        self.rpn_config = model_config['rpn_config']
        self.fc_output_layer_config = model_config['fc_output_layer_config']

        # assigner
        self.target_assigner = TargetAssigner(
            model_config['target_assigner_config'])

        # coder
        self.bbox_coder = self.target_assigner.bbox_coder

        # sampler
        self.sampler = BalancedSampler(model_config['sampler_config'])

        self.reduce = True

        self.profiler = Profiler()

        self.anchor_projector = AnchorProjector()

    def loss(self, prediction_dict, feed_dict):
        # loss for cls

        loss_dict = {}
        loss_dict.update(self.rpn_model.loss(prediction_dict, feed_dict))
        gt_boxes_3d = feed_dict['gt_boxes_3d']
        top_proposals = prediction_dict['proposals_batch'].unsqueeze(0)

        # increate batch dim
        gt_boxes_bev = self.anchor_projector.project_to_bev(
            gt_boxes_3d[0], self.bev_extents).unsqueeze(0)
        bev_proposal_boxes = self.anchor_projector.project_to_bev(
            top_proposals[0], self.bev_extents).unsqueeze(0)

        #################################
        # target assigner
        ################################
        rcnn_cls_targets, rcnn_reg_targets, \
            rcnn_cls_weights, rcnn_reg_weights = \
                self.target_assigner.assign(
                    bev_proposal_boxes,
                    gt_boxes_bev,
                    top_proposals,
                    gt_boxes_3d,
                    gt_labels=None)

        ################################
        # subsample
        ################################
        pos_indicator = rcnn_reg_weights > 0
        indicator = rcnn_cls_weights > 0

        rcnn_cls_probs = prediction_dict['all_cls_softmax'][:, 1]
        cls_criterion = rcnn_cls_probs

        batch_sampled_mask = self.sampler.subsample_batch(
            self.rcnn_batch_size,
            pos_indicator,
            criterion=cls_criterion,
            indicator=indicator)
        batch_sampled_mask = batch_sampled_mask.type_as(rcnn_cls_weights)
        rcnn_cls_weights = rcnn_cls_weights * batch_sampled_mask
        rcnn_reg_weights = rcnn_reg_weights * batch_sampled_mask
        num_cls_coeff = (rcnn_cls_weights > 0).sum(dim=1)
        num_reg_coeff = (rcnn_reg_weights > 0).sum(dim=1)
        # check
        #  assert num_cls_coeff, 'bug happens'
        #  assert num_reg_coeff, 'bug happens'
        if num_cls_coeff == 0:
            num_cls_coeff = torch.ones([]).type_as(num_cls_coeff)
        if num_reg_coeff == 0:
            num_reg_coeff = torch.ones([]).type_as(num_reg_coeff)

        # cls loss
        rcnn_cls_score = prediction_dict['all_cls_logits']
        rcnn_cls_loss = self.rcnn_cls_loss(
            rcnn_cls_score.view(-1, 2), rcnn_cls_targets.view(-1))
        rcnn_cls_loss = rcnn_cls_loss.view_as(rcnn_cls_weights)
        rcnn_cls_loss = rcnn_cls_loss * rcnn_cls_weights
        rcnn_cls_loss = rcnn_cls_loss.sum(dim=1) / num_cls_coeff.float()

        # bbox loss
        rcnn_bbox_preds = prediction_dict['all_reg_3d'].unsqueeze(0)
        rcnn_reg_loss = self.rcnn_bbox_loss(rcnn_bbox_preds, rcnn_reg_targets)
        rcnn_reg_loss = rcnn_reg_loss * rcnn_reg_weights.unsqueeze(-1)
        rcnn_reg_loss = rcnn_reg_loss.view(rcnn_reg_loss.shape[0], -1).sum(
            dim=1) / num_reg_coeff.float()

        loss_dict['rcnn_cls_loss'] = rcnn_cls_loss
        loss_dict['rcnn_bbox_loss'] = rcnn_reg_loss

        prediction_dict['rcnn_reg_weights'] = rcnn_reg_weights[
            batch_sampled_mask.byte()]

        fake_match = self.target_assigner.analyzer.match
        num_gt = feed_dict['gt_labels'].numel()
        self.target_assigner.analyzer.analyze_ap(
            fake_match, rcnn_cls_probs, num_gt, thresh=0.5)
        return loss_dict
