# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import Model
from core.models.double_iou_rpn_model import RPNModel
from core.models.focal_loss import FocalLoss
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.psroi_pooling.modules.psroi_pool import PSRoIPool

from core.filler import Filler
from core.double_iou_target_assigner import TargetAssigner
from core.samplers.hard_negative_sampler import HardNegativeSampler
from core.samplers.balanced_sampler import BalancedSampler
from core.models.feature_extractors.resnet import ResNetFeatureExtractor
from core.samplers.detection_sampler import DetectionSampler

from utils import box_ops
import copy

import functools


class PostCLSIOUFasterRCNN(Model):
    def forward(self, feed_dict):
        # some pre forward hook
        self.clean_stats()

        prediction_dict = {}

        ################################
        # first stage
        ################################
        # base model
        base_feat = self.feature_extractor.first_stage_feature(
            feed_dict['img'])
        feed_dict.update({'base_feat': base_feat})
        self.add_feat('base_feat', base_feat)

        # rpn model
        prediction_dict.update(self.rpn_model.forward(feed_dict))

        #####################################
        # second stage(bbox regression)
        #####################################
        # pre subsample for reduce consume of memory
        if self.training and self.enable_reg:
            # append gt
            if self.use_gt:
                prediction_dict['rois_batch'] = self.append_gt(
                    prediction_dict['rois_batch'], feed_dict['gt_boxes'])
            stats = self.pre_subsample(prediction_dict, feed_dict)
            # rois stats
            self.stats.update(stats)
        rois_batch = prediction_dict['rois_batch']

        # note here base_feat (N,C,H,W),rois_batch (N,num_proposals,5)
        pooled_feat = self.rcnn_pooling(base_feat, rois_batch.view(-1, 5))

        # although it must be true
        #  if self.enable_reg:
        # shape(N,C,1,1)
        pooled_feat_reg = self.feature_extractor.second_stage_feature(
            pooled_feat)

        pooled_feat_reg = pooled_feat_reg.mean(3).mean(2)
        rcnn_bbox_preds = self.rcnn_bbox_pred(pooled_feat_reg)
        prediction_dict['rcnn_bbox_preds'] = rcnn_bbox_preds

        # used for tracking
        proposals_order = prediction_dict['proposals_order']
        prediction_dict['second_rpn_anchors'] = prediction_dict['anchors'][
            proposals_order]
        prediction_dict['second_rpn_cls_probs'] = prediction_dict[
            'rpn_cls_probs'][0][proposals_order]

        ###########################################
        # third stage(predict scores of final bbox)
        ###########################################

        # decode rcnn bbox, generate rcnn rois batch
        pred_boxes = self.bbox_coder.decode_batch(
            rcnn_bbox_preds.view(1, -1, 4), rois_batch[:, :, 1:5])
        rcnn_rois_batch = torch.zeros_like(rois_batch)
        rcnn_rois_batch[:, :, 1:5] = pred_boxes.detach()
        prediction_dict['rcnn_rois_batch'] = rcnn_rois_batch

        if self.training and self.use_gt:
            # append gt
            rcnn_rois_batch = self.append_gt(rcnn_rois_batch,
                                             feed_dict['gt_boxes'])
            prediction_dict['rcnn_rois_batch'] = rcnn_rois_batch

        if self.enable_cls:
            if self.training:
                rcnn_stats = self.pre_subsample(
                    prediction_dict, feed_dict, stage='rcnn')
                # rcnn stats
                self.rcnn_stats.update(rcnn_stats)

            # rois after subsample
            pred_rois = prediction_dict['rcnn_rois_batch']
            pooled_feat_cls = self.rcnn_pooling(base_feat,
                                                pred_rois.view(-1, 5))
            pooled_feat_cls = self.feature_extractor.third_stage_feature(
                pooled_feat_cls.detach())

            # shape(N,C)
            pooled_feat_cls = pooled_feat_cls.mean(3).mean(2)
            rcnn_cls_scores = self.rcnn_cls_pred(pooled_feat_cls)
            rcnn_cls_probs = F.softmax(rcnn_cls_scores, dim=1)

            prediction_dict['rcnn_cls_probs'] = rcnn_cls_probs
            prediction_dict['rcnn_cls_scores'] = rcnn_cls_scores

        ###################################
        # stats
        ###################################
        # import ipdb
        # ipdb.set_trace()
        if not self.training or (self.enable_track_rois and
                                 not self.enable_reg):
            # when enable reg, skip it,
            stats = self.target_assigner.assign(rois_batch[:, :, 1:],
                                                feed_dict['gt_boxes'],
                                                feed_dict['gt_labels'])[-1]
            self.stats.update(stats)

        if not self.training or (self.enable_track_rcnn_rois and
                                 not self.enable_cls):
            # when enable cls, skip it
            stats = self.target_assigner.assign(rcnn_rois_batch[:, :, 1:],
                                                feed_dict['gt_boxes'],
                                                feed_dict['gt_labels'])[-1]
            self.rcnn_stats.update(stats)

        # analysis ap
        # when enable cls, otherwise it is no sense
        if self.training and self.enable_cls:
            rcnn_cls_probs = prediction_dict['rcnn_cls_probs']
            num_gt = feed_dict['gt_labels'].numel()
            fake_match = self.rcnn_stats['match']
            stats = self.target_assigner.analyzer.analyze_ap(
                fake_match, rcnn_cls_probs[:, 1], num_gt, thresh=0.5)
            # collect stats
            self.rcnn_stats.update(stats)

        return prediction_dict

    def append_gt(self, rois_batch, gt_boxes):
        ################################
        # append gt_boxes to rois_batch for losses
        ################################
        # may be some bugs here
        gt_boxes_append = torch.zeros(gt_boxes.shape[0], gt_boxes.shape[1],
                                      5).type_as(gt_boxes)
        gt_boxes_append[:, :, 1:5] = gt_boxes[:, :, :4]
        # cat gt_boxes to rois_batch
        rois_batch = torch.cat([rois_batch, gt_boxes_append], dim=1)
        return rois_batch

    def init_weights(self):
        # submodule init weights
        self.feature_extractor.init_weights()
        self.rpn_model.init_weights()

        Filler.normal_init(self.rcnn_cls_pred, 0, 0.01, self.truncated)
        Filler.normal_init(self.rcnn_bbox_pred, 0, 0.001, self.truncated)

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
            self.rcnn_cls_loss = FocalLoss(2, alpha=0.25, gamma=2)
        else:
            self.rcnn_cls_loss = functools.partial(
                F.cross_entropy, reduce=False)

        self.rcnn_bbox_loss = nn.modules.SmoothL1Loss(reduce=False)

    def init_param(self, model_config):
        classes = model_config['classes']
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = model_config['class_agnostic']
        self.pooling_size = model_config['pooling_size']
        self.pooling_mode = model_config['pooling_mode']
        self.crop_resize_with_max_pool = model_config[
            'crop_resize_with_max_pool']
        self.truncated = model_config['truncated']

        self.use_focal_loss = model_config['use_focal_loss']
        self.subsample_twice = model_config['subsample_twice']
        self.rcnn_batch_size = model_config['rcnn_batch_size']

        # some submodule config
        self.feature_extractor_config = model_config['feature_extractor_config']
        self.rpn_config = model_config['rpn_config']

        # assigner
        self.target_assigner = TargetAssigner(
            model_config['target_assigner_config'])

        # bbox_coder
        self.bbox_coder = self.target_assigner.bbox_coder

        # similarity
        self.similarity_calc = self.target_assigner.similarity_calc

        # sampler
        self.sampler = BalancedSampler(model_config['sampler_config'])

        # self.reduce = model_config.get('reduce')
        self.reduce = True

        # optimize cls
        self.enable_cls = True

        # optimize reg
        self.enable_reg = False

        # cal iou
        self.enable_iou = False

        # track good rois
        self.enable_track_rois = True
        self.enable_track_rcnn_rois = True

        # eval the final bbox
        self.enable_eval_final_bbox = True

        # use gt
        self.use_gt = False

        # if self.enable_eval_final_bbox:
        self.subsample = False

    def clean_stats(self):
        # rois bbox
        self.stats = {
            'num_det': 1,
            'num_tp': 0,
            'matched_thresh': 0,
            'recall_thresh': 0,
            'match': None
        }

        # rcnn bbox(final bbox)
        self.rcnn_stats = {
            'num_det': 1,
            'num_tp': 0,
            'matched_thresh': 0,
            'recall_thresh': 0,
            'match': None
        }

    def pre_subsample(self, prediction_dict, feed_dict, stage='rpn'):
        if stage == 'rpn':
            rois_name = 'rois_batch'
        else:
            rois_name = 'rcnn_rois_batch'

        rois_batch = prediction_dict[rois_name]

        gt_boxes = feed_dict['gt_boxes']
        gt_labels = feed_dict['gt_labels']

        # append gt
        # rois_batch = self.append_gt(rois_batch, gt_boxes)

        ##########################
        # assigner
        ##########################
        # import ipdb
        # ipdb.set_trace()
        rcnn_cls_targets, rcnn_reg_targets, rcnn_cls_weights,\
            rcnn_reg_weights, stats, iou = self.target_assigner.assign(
            rois_batch[:, :, 1:], gt_boxes, gt_labels, ret_iou=True)

        ##########################
        # subsampler
        ##########################
        if self.subsample:
            cls_criterion = None

            if self.enable_reg:
                # used for reg training
                pos_indicator = rcnn_reg_weights > 0
                indicator = None
            elif self.enable_cls:
                # used for cls training
                pos_indicator = rcnn_cls_targets > 0
                indicator = rcnn_cls_weights > 0
            else:
                raise ValueError(
                    "please check enable reg and enable cls again")

            # subsample from all
            # shape (N,M)
            batch_sampled_mask = self.sampler.subsample_batch(
                self.rcnn_batch_size,
                pos_indicator,
                indicator=indicator,
                criterion=cls_criterion)
        else:
            batch_sampled_mask = torch.ones_like(rcnn_cls_weights > 0)

        if self.enable_cls:
            rcnn_cls_weights = rcnn_cls_weights[batch_sampled_mask]
            num_cls_coeff = (rcnn_cls_weights > 0).sum(dim=-1)
            assert num_cls_coeff, 'bug happens'
            prediction_dict[
                'rcnn_cls_weights'] = rcnn_cls_weights / num_cls_coeff.float()

        # used for retriving statistic
        prediction_dict['rcnn_cls_targets'] = rcnn_cls_targets[
            batch_sampled_mask]

        # used for fg/bg
        rcnn_reg_weights = rcnn_reg_weights[batch_sampled_mask]
        num_reg_coeff = (rcnn_reg_weights > 0).sum(dim=-1)
        num_reg_coeff = torch.max(num_reg_coeff,
                                  torch.ones_like(num_reg_coeff))
        # import ipdb
        # ipdb.set_trace()
        # assert num_reg_coeff, 'bug happens'
        prediction_dict[
            'rcnn_reg_weights'] = rcnn_reg_weights / num_reg_coeff.float()

        if self.enable_reg:
            prediction_dict['rcnn_reg_targets'] = rcnn_reg_targets[
                batch_sampled_mask]

        # here use rcnn_target_assigner for final bbox pred
        stats['match'] = stats['match'][batch_sampled_mask]

        # update rois_batch
        prediction_dict[rois_name] = rois_batch[batch_sampled_mask].view(
            rois_batch.shape[0], -1, 5)
        return stats

    def loss(self, prediction_dict, feed_dict):
        """
        assign proposals label and subsample from them
        Then calculate loss
        """
        loss_dict = {}

        # submodule loss

        # add rcnn_cls_targets to get the statics of rpn
        #  loss_dict['rcnn_cls_targets'] = rcnn_cls_targets

        if self.enable_cls:
            # targets and weights
            rcnn_cls_weights = prediction_dict['rcnn_cls_weights']

            rcnn_cls_targets = prediction_dict['rcnn_cls_targets']
            # classification loss
            rcnn_cls_scores = prediction_dict['rcnn_cls_scores']
            rcnn_cls_loss = self.rcnn_cls_loss(rcnn_cls_scores,
                                               rcnn_cls_targets)
            rcnn_cls_loss *= rcnn_cls_weights
            rcnn_cls_loss = rcnn_cls_loss.sum(dim=-1)

            loss_dict['rcnn_cls_loss'] = rcnn_cls_loss

        if self.enable_reg:
            loss_dict.update(self.rpn_model.loss(prediction_dict, feed_dict))

            rcnn_reg_weights = prediction_dict['rcnn_reg_weights']
            rcnn_reg_targets = prediction_dict['rcnn_reg_targets']

            # bounding box regression L1 loss
            rcnn_bbox_preds = prediction_dict['rcnn_bbox_preds']
            rcnn_bbox_loss = self.rcnn_bbox_loss(rcnn_bbox_preds,
                                                 rcnn_reg_targets).sum(dim=-1)
            rcnn_bbox_loss *= rcnn_reg_weights
            rcnn_bbox_loss = rcnn_bbox_loss.sum(dim=-1)

            # loss weights has no gradients
            loss_dict['rcnn_bbox_loss'] = rcnn_bbox_loss

        return loss_dict
