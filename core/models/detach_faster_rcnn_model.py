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
from core.target_assigner import TargetAssigner
from core.samplers.hard_negative_sampler import HardNegativeSampler
from core.samplers.balanced_sampler import BalancedSampler
from core.models.feature_extractors.resnet import ResNetFeatureExtractor
from core.models.feature_extractors.mobilenet import MobileNetFeatureExtractor
from core.samplers.detection_sampler import DetectionSampler
from builder import feature_extractors_builder

import functools


class DetachFasterRCNN(Model):
    def collect_intermedia_layers(self, img):
        feat2 = self.feature_extractor.first_stage_feature[:-1](img)
        feat3 = self.feature_extractor.first_stage_feature[-1](feat2)

        end_points = {'feat2': feat2, 'feat3': feat3}
        return feat3, end_points

    def caroi_pooling(self, all_feats, rois_batch, out_channels):
        # import ipdb
        # ipdb.set_trace()
        pooled_feats = []
        for feat_name in all_feats:
            if feat_name == 'feat2':
                pooled_feat = self.rcnn_pooling2(all_feats[feat_name],
                                                 rois_batch)
            else:
                continue
            # else:
            # pooled_feat = self.rcnn_pooling(all_feats[feat_name],
            # rois_batch)
            pooled_feats.append(pooled_feat)
        pooled_feats = torch.cat(pooled_feats, dim=1)
        if pooled_feats.shape[1] != out_channels:
            # add 1x1 conv
            pooled_feats = self.reduce_pooling(pooled_feats)
        return pooled_feats

    def forward(self, feed_dict):

        prediction_dict = {}

        # base model
        base_feat, all_feats = self.collect_intermedia_layers(feed_dict['img'])
        feed_dict.update({'base_feat': base_feat})
        self.add_feat('feat2', all_feats['feat2'])
        self.add_feat('base_feat', base_feat)
        # batch_size = base_feat.shape[0]

        # rpn model
        prediction_dict.update(self.rpn_model.forward(feed_dict))

        # proposals = prediction_dict['proposals_batch']
        # shape(N,num_proposals,5)
        # pre subsample for reduce consume of memory
        if self.training:
            self.pre_subsample(prediction_dict, feed_dict)
        rois_batch = prediction_dict['rois_batch']

        # note here base_feat (N,C,H,W),rois_batch (N,num_proposals,5)
        # pooled_feat = self.caroi_pooling(
        # all_feats, rois_batch.view(-1, 5), out_channels=1024)
        pooled_feat = self.rcnn_pooling(base_feat, rois_batch.view(-1, 5))

        # shape(N,C,1,1)
        # pooled_feat = self.feature_extractor.second_stage_feature(pooled_feat)
        pooled_feat_reg = self.feature_extractor.second_stage_feature(
            pooled_feat)
        pooled_feat_cls = self.feature_extractor.third_stage_feature(
            pooled_feat.detach())
        self.add_feat('base_feat', base_feat)

        pooled_feat_cls = pooled_feat_cls.mean(3).mean(2)
        pooled_feat_reg = pooled_feat_reg.mean(3).mean(2)
        rcnn_cls_scores = self.rcnn_cls_pred(pooled_feat_cls)
        rcnn_bbox_preds = self.rcnn_bbox_pred(pooled_feat_reg)
        rcnn_cls_probs = F.softmax(rcnn_cls_scores, dim=1)

        prediction_dict['rcnn_cls_probs'] = rcnn_cls_probs
        prediction_dict['rcnn_bbox_preds'] = rcnn_bbox_preds
        prediction_dict['rcnn_cls_scores'] = rcnn_cls_scores

        # used for track
        proposals_order = prediction_dict['proposals_order']
        prediction_dict['second_rpn_anchors'] = prediction_dict['anchors'][
            proposals_order]

        return prediction_dict

    def generate_channel_attention(self, feat):
        return feat.mean(3, keepdim=True).mean(2, keepdim=True)

    def generate_spatial_attention(self, feat):
        return self.spatial_attention(feat)

    def init_weights(self):
        # submodule init weights
        self.feature_extractor.init_weights()
        self.rpn_model.init_weights()

        Filler.normal_init(self.rcnn_cls_pred, 0, 0.01, self.truncated)
        Filler.normal_init(self.rcnn_bbox_pred, 0, 0.001, self.truncated)

    def init_modules(self):
        self.feature_extractor = feature_extractors_builder.build(
            self.feature_extractor_config)
        # self.feature_extractor = ResNetFeatureExtractor(
        # self.feature_extractor_config)
        # self.feature_extractor = MobileNetFeatureExtractor(
        # self.feature_extractor_config)
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
        self.rcnn_cls_pred = nn.Linear(self.ndin, self.n_classes)
        if self.class_agnostic:
            self.rcnn_bbox_pred = nn.Linear(self.ndin, 4)
            # self.rcnn_bbox_pred = nn.Conv2d(2048,4,3,1,1)
        else:
            self.rcnn_bbox_pred = nn.Linear(self.ndin, 4 * self.n_classes)

        # loss module
        if self.use_focal_loss:
            self.rcnn_cls_loss = FocalLoss(
                2, gamma=2, alpha=0.2, auto_alpha=False)
        else:
            self.rcnn_cls_loss = functools.partial(
                F.cross_entropy, reduce=False)

        self.rcnn_bbox_loss = nn.modules.SmoothL1Loss(reduce=False)

        # attention
        if self.use_self_attention:
            self.spatial_attention = nn.Conv2d(self.ndin, 1, 3, 1, 1)

        self.rcnn_pooling2 = RoIAlignAvg(self.pooling_size, self.pooling_size,
                                         1.0 / 8.0)
        self.reduce_pooling = nn.Sequential(
            nn.Conv2d(512, 1024, 1, 1, 0), nn.ReLU())

    def init_param(self, model_config):
        if model_config.get('din'):
            self.ndin = model_config['din']
        else:
            self.ndin = 512
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
        self.use_self_attention = model_config.get('use_self_attention')

        # some submodule config
        self.feature_extractor_config = model_config['feature_extractor_config']
        self.rpn_config = model_config['rpn_config']

        # assigner
        self.target_assigner = TargetAssigner(
            model_config['target_assigner_config'])

        # sampler
        self.sampler = BalancedSampler(model_config['sampler_config'])

    def pre_subsample(self, prediction_dict, feed_dict):
        rois_batch = prediction_dict['rois_batch']
        gt_boxes = feed_dict['gt_boxes']
        gt_labels = feed_dict['gt_labels']

        ##########################
        # assigner
        ##########################
        #  import ipdb
        #  ipdb.set_trace()
        rcnn_cls_targets, rcnn_reg_targets, rcnn_cls_weights, rcnn_reg_weights = self.target_assigner.assign(
            rois_batch[:, :, 1:], gt_boxes, gt_labels)

        ##########################
        # subsampler
        ##########################
        cls_criterion = None
        pos_indicator = rcnn_reg_weights > 0
        indicator = rcnn_cls_weights > 0

        # subsample from all
        # shape (N,M)
        batch_sampled_mask = self.sampler.subsample_batch(
            self.rcnn_batch_size,
            pos_indicator,
            indicator=indicator,
            criterion=cls_criterion)
        rcnn_cls_weights = rcnn_cls_weights[batch_sampled_mask]
        rcnn_reg_weights = rcnn_reg_weights[batch_sampled_mask]
        num_cls_coeff = (rcnn_cls_weights > 0).sum(dim=-1)
        num_reg_coeff = (rcnn_reg_weights > 0).sum(dim=-1)
        # check
        assert num_cls_coeff, 'bug happens'
        assert num_reg_coeff, 'bug happens'

        prediction_dict[
            'rcnn_cls_weights'] = rcnn_cls_weights / num_cls_coeff.float()
        prediction_dict[
            'rcnn_reg_weights'] = rcnn_reg_weights / num_reg_coeff.float()
        prediction_dict['rcnn_cls_targets'] = rcnn_cls_targets[
            batch_sampled_mask]
        prediction_dict['rcnn_reg_targets'] = rcnn_reg_targets[
            batch_sampled_mask]
        prediction_dict['fake_match'] = self.target_assigner.analyzer.match[
            batch_sampled_mask]

        # update rois_batch
        prediction_dict['rois_batch'] = rois_batch[batch_sampled_mask].view(
            rois_batch.shape[0], -1, 5)

        if not self.training:
            # used for track
            proposals_order = prediction_dict['proposals_order']

            prediction_dict['proposals_order'] = proposals_order[
                batch_sampled_mask]

    def loss(self, prediction_dict, feed_dict):
        """
        assign proposals label and subsample from them
        Then calculate loss
        """
        loss_dict = {}

        # submodule loss
        loss_dict.update(self.rpn_model.loss(prediction_dict, feed_dict))

        # targets and weights
        rcnn_cls_weights = prediction_dict['rcnn_cls_weights']
        rcnn_reg_weights = prediction_dict['rcnn_reg_weights']

        rcnn_cls_targets = prediction_dict['rcnn_cls_targets']
        rcnn_reg_targets = prediction_dict['rcnn_reg_targets']

        # classification loss
        rcnn_cls_scores = prediction_dict['rcnn_cls_scores']
        rcnn_cls_loss = self.rcnn_cls_loss(rcnn_cls_scores, rcnn_cls_targets)
        rcnn_cls_loss *= rcnn_cls_weights
        rcnn_cls_loss = rcnn_cls_loss.sum(dim=-1)

        # bounding box regression L1 loss
        rcnn_bbox_preds = prediction_dict['rcnn_bbox_preds']
        rcnn_bbox_loss = self.rcnn_bbox_loss(rcnn_bbox_preds,
                                             rcnn_reg_targets).sum(dim=-1)
        rcnn_bbox_loss *= rcnn_reg_weights
        # rcnn_bbox_loss *= rcnn_reg_weights
        rcnn_bbox_loss = rcnn_bbox_loss.sum(dim=-1)

        # loss weights has no gradients
        loss_dict['rcnn_cls_loss'] = rcnn_cls_loss
        loss_dict['rcnn_bbox_loss'] = rcnn_bbox_loss

        # add rcnn_cls_targets to get the statics of rpn
        # loss_dict['rcnn_cls_targets'] = rcnn_cls_targets

        # analysis ap
        rcnn_cls_probs = prediction_dict['rcnn_cls_probs']
        num_gt = feed_dict['gt_labels'].numel()
        fake_match = prediction_dict['fake_match']
        self.target_assigner.analyzer.analyze_ap(
            fake_match, rcnn_cls_probs[:, 1], num_gt, thresh=0.5)

        return loss_dict
