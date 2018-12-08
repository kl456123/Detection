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

import functools


class DetachDoubleIOUFasterRCNN(Model):
    def forward(self, feed_dict):

        prediction_dict = {}

        # base model
        base_feat = self.feature_extractor.first_stage_feature(
            feed_dict['img'])
        feed_dict.update({'base_feat': base_feat})
        self.add_feat('base_feat', base_feat)

        # rpn model
        prediction_dict.update(self.rpn_model.forward(feed_dict))

        # pre subsample for reduce consume of memory
        if self.training:
            self.pre_subsample(prediction_dict, feed_dict)
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

        if self.enable_cls:
            pooled_feat_cls = self.feature_extractor.third_stage_feature(
                pooled_feat.detach())

            # shape(N,C)
            pooled_feat_cls = pooled_feat_cls.mean(3).mean(2)
            rcnn_cls_scores = self.rcnn_cls_pred(pooled_feat_cls)
            rcnn_cls_probs = F.softmax(rcnn_cls_scores, dim=1)

            prediction_dict['rcnn_cls_probs'] = rcnn_cls_probs
            prediction_dict['rcnn_cls_scores'] = rcnn_cls_scores

        # used for track
        proposals_order = prediction_dict['proposals_order']
        prediction_dict['second_rpn_anchors'] = prediction_dict['anchors'][
            proposals_order]
        prediction_dict['second_rpn_cls_probs'] = prediction_dict[
            'rpn_cls_probs'][0][proposals_order]

        if not self.training and self.enable_iou:
            # calculate fake iou as final score,of course use scores to filter bg
            pred_boxes = self.bbox_coder.decode_batch(
                rcnn_bbox_preds.view(1, -1, 4), rois_batch[:, :, 1:5])
            iou_matrix = box_ops.iou(pred_boxes, rois_batch[:, :, 1:5])[0]
            iou_matrix[rcnn_cls_probs[:, 1] < 0.5] = 0
            rcnn_cls_probs[:, 1] = iou_matrix
            prediction_dict['rcnn_cls_probs'] = rcnn_cls_probs

        if not self.training and self.enable_track_rois:
            self.target_assigner.assign(rois_batch[:, :, 1:],
                                        feed_dict['gt_boxes'],
                                        feed_dict['gt_labels'])

        return prediction_dict

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
            self.rcnn_cls_loss = FocalLoss(2, alpha=0.25)
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
        self.enable_cls = False

        # optimize reg
        self.enable_reg = True

        # cal iou
        self.enable_iou = False

        # track good rois
        self.enable_track_rois = False

    def pre_subsample(self, prediction_dict, feed_dict):
        rois_batch = prediction_dict['rois_batch']
        gt_boxes = feed_dict['gt_boxes']
        gt_labels = feed_dict['gt_labels']

        ##########################
        # assigner
        ##########################
        rcnn_cls_targets, rcnn_reg_targets, rcnn_cls_weights, rcnn_reg_weights = self.target_assigner.assign(
            rois_batch[:, :, 1:], gt_boxes, gt_labels)

        ##########################
        # subsampler
        ##########################
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
            raise ValueError("please check enable reg and enable cls again")

        # subsample from all
        # shape (N,M)
        batch_sampled_mask = self.sampler.subsample_batch(
            self.rcnn_batch_size,
            pos_indicator,
            indicator=indicator,
            criterion=cls_criterion)

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
        assert num_reg_coeff, 'bug happens'
        prediction_dict[
            'rcnn_reg_weights'] = rcnn_reg_weights / num_reg_coeff.float()

        if self.enable_reg:
            prediction_dict['rcnn_reg_targets'] = rcnn_reg_targets[
                batch_sampled_mask]

        prediction_dict['fake_match'] = self.target_assigner.analyzer.match[
            batch_sampled_mask]

        # update rois_batch
        prediction_dict['rois_batch'] = rois_batch[batch_sampled_mask].view(
            rois_batch.shape[0], -1, 5)

    def loss(self, prediction_dict, feed_dict):
        """
        assign proposals label and subsample from them
        Then calculate loss
        """
        loss_dict = {}

        # submodule loss
        loss_dict.update(self.rpn_model.loss(prediction_dict, feed_dict))

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

        # analysis ap
        # when enable cls,otherwise it is no sense
        if self.enable_cls:
            rcnn_cls_probs = prediction_dict['rcnn_cls_probs']
            num_gt = feed_dict['gt_labels'].numel()
            fake_match = prediction_dict['fake_match']
            self.target_assigner.analyzer.analyze_ap(
                fake_match, rcnn_cls_probs[:, 1], num_gt, thresh=0.5)

        return loss_dict
