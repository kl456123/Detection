# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import Model
from core.models.rpn_model import RPNModel
from core.models.focal_loss import FocalLoss
from model.roi_align.modules.roi_align import RoIAlignAvg

from core.filler import Filler
from core.target_assigner import TargetAssigner
from core.samplers.hard_negative_sampler import HardNegativeSampler
from core.samplers.balanced_sampler import BalancedSampler
from core.models.feature_extractor_model import FeatureExtractor

from lib.model.utils.net_utils import _smooth_l1_loss
from lib.model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer as ProposalTarget
from lib.model.rpn.proposal_target_layer_cascade_org import _ProposalTargetLayer as ProposalTargetOrg


class FasterRCNN(Model):
    def forward_old(self, feed_dict):
        gt_boxes = feed_dict['gt_boxes']
        gt_labels = feed_dict['gt_labels']
        gt_boxes = torch.cat([gt_boxes, gt_labels.unsqueeze(2).float()], dim=2)
        batch_size = gt_boxes.shape[0]
        prediction_dict = {}
        # num_boxes = feed_dict['num_boxes']

        # base model
        base_feat = self.feature_extractor.first_stage_feature(
            feed_dict['img'])
        feed_dict.update({'base_feat': base_feat})

        # rpn model
        prediction_dict.update(self.rpn_model.forward(feed_dict))

        # proposals = prediction_dict['proposals_batch']
        # shape(N,num_proposals,5)
        rois_batch = prediction_dict['rois_batch']

        if self.training:
            roi_data = self.RCNN_proposal_target(rois_batch, gt_boxes)
            rois_batch, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = rois_label.view(-1).long()
            rois_target = rois_target.view(-1, rois_target.size(2))
            rois_inside_ws = rois_inside_ws.view(-1, rois_inside_ws.size(2))
            rois_outside_ws = rois_outside_ws.view(-1, rois_outside_ws.size(2))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            # rpn_loss_cls = 0
            # rpn_loss_bbox = 0

        # do roi pooling based on predicted rois

        pooled_feat = self.rcnn_pooling(base_feat, rois_batch.view(-1, 5))

        # feed pooled features to top model
        # shape(N,C,1,1)
        pooled_feat = self.feature_extractor.second_stage_feature(pooled_feat)
        # shape(N,C)
        pooled_feat = pooled_feat.mean(3).mean(2)
        # pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.rcnn_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(
                bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(
                bbox_pred_view, 1,
                rois_label.view(rois_label.size(0), 1, 1).expand(
                    rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute object classification probability
        cls_score = self.rcnn_cls_pred(pooled_feat)
        cls_prob = F.softmax(cls_score)

        rcnn_loss_cls = 0
        rcnn_loss_bbox = 0

        if self.training:
            # classification loss
            rcnn_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            rcnn_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target,
                                             rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob.view(batch_size, rois_batch.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois_batch.size(1), -1)

        proposals_order = prediction_dict['proposals_order']
        prediction_dict['second_rpn_anchors'] = prediction_dict['anchors'][0][
            proposals_order]

        prediction_dict.update({
            'rois_batch': rois_batch,
            'rcnn_cls_targets': rois_label,
            'rcnn_cls_probs': cls_prob,
            'rcnn_bbox_preds': bbox_pred,
            'rcnn_cls_loss': rcnn_loss_cls,
            'rcnn_bbox_loss': rcnn_loss_bbox
        })

        return prediction_dict

    def forward(self, feed_dict):

        prediction_dict = {}

        # base model
        base_feat = self.feature_extractor.first_stage_feature(
            feed_dict['img'])
        feed_dict.update({'base_feat': base_feat})
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
        pooled_feat = self.rcnn_pooling(base_feat, rois_batch.view(-1, 5))

        # shape(N,C,1,1)
        pooled_feat = self.feature_extractor.second_stage_feature(pooled_feat)
        # shape(N,C)
        pooled_feat = pooled_feat.mean(3).mean(2)

        rcnn_bbox_preds = self.rcnn_bbox_pred(pooled_feat)
        rcnn_cls_scores = self.rcnn_cls_pred(pooled_feat)

        rcnn_cls_probs = F.softmax(rcnn_cls_scores, dim=1)

        prediction_dict['rcnn_cls_probs'] = rcnn_cls_probs
        prediction_dict['rcnn_bbox_preds'] = rcnn_bbox_preds
        prediction_dict['rcnn_cls_scores'] = rcnn_cls_scores

        # used for track
        proposals_order = prediction_dict['proposals_order']
        prediction_dict['second_rpn_anchors'] = prediction_dict['anchors'][0][
            proposals_order]

        return prediction_dict

    def init_weights(self):
        # submodule init weights
        self.feature_extractor.init_weights()
        self.rpn_model.init_weights()

        Filler.normal_init(self.rcnn_cls_pred, 0, 0.01, self.truncated)
        Filler.normal_init(self.rcnn_bbox_pred, 0, 0.001, self.truncated)

    def init_modules(self):
        self.feature_extractor = FeatureExtractor(
            self.feature_extractor_config)
        self.rpn_model = RPNModel(self.rpn_config)
        self.rcnn_pooling = RoIAlignAvg(self.pooling_size, self.pooling_size,
                                        1.0 / 16.0)
        self.rcnn_cls_pred = nn.Linear(2048, self.n_classes)
        if self.class_agnostic:
            self.rcnn_bbox_pred = nn.Linear(2048, 4)
        else:
            self.rcnn_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

        # loss module
        if self.use_focal_loss:
            self.rcnn_cls_loss = FocalLoss(2)
        else:
            self.rcnn_cls_loss = F.cross_entropy

        self.rcnn_bbox_loss = nn.modules.SmoothL1Loss(reduce=False)

        self.proposal_target_layer_config = {
            "use_focal_loss": self.use_focal_loss,
            "nclasses": self.n_classes,
            "bbox_normalize_stds":
            self.target_assigner.bbox_coder.bbox_normalize_stds,
            "bbox_normalize_means":
            self.target_assigner.bbox_coder.bbox_normalize_means,
            "bbox_inside_weights": [1.0, 1.0, 1.0, 1.0],
            "batch_size": self.rcnn_batch_size,
            "fg_fraction": self.sampler.fg_fraction,
            "bbox_normalize_targets_precomputed":
            self.target_assigner.bbox_coder.bbox_normalize_targets_precomputed,
            "fg_thresh": self.target_assigner.fg_thresh,
            "bg_thresh": self.target_assigner.bg_thresh,
            "bg_thresh_lo": 0.0
        }
        use_org = True
        if use_org:

            self.RCNN_proposal_target = ProposalTargetOrg(
                self.proposal_target_layer_config)
        else:
            self.RCNN_proposal_target = ProposalTarget(
                self.proposal_target_layer_config)

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

        # sampler
        # self.sampler = HardNegativeSampler(model_config['sampler_config'])
        self.sampler = BalancedSampler(model_config['sampler_config'])

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
        pos_indicator = rcnn_cls_targets > 0
        critation = rcnn_cls_weights > 0

        # subsample from all
        # shape (N,M)
        batch_sampled_mask = self.sampler.subsample_batch(
            self.rcnn_batch_size, pos_indicator, criterion=critation)
        rcnn_cls_weights = rcnn_cls_weights[batch_sampled_mask]
        rcnn_reg_weights = rcnn_reg_weights[batch_sampled_mask]
        num_cls_coeff = rcnn_cls_weights.type(torch.cuda.ByteTensor).sum(
            dim=-1)
        num_reg_coeff = rcnn_reg_weights.type(torch.cuda.ByteTensor).sum(
            dim=-1)

        prediction_dict[
            'rcnn_cls_weights'] = rcnn_cls_weights / num_cls_coeff.float()
        prediction_dict[
            'rcnn_reg_weights'] = rcnn_reg_weights / num_reg_coeff.float()
        prediction_dict['rcnn_cls_targets'] = rcnn_cls_targets[
            batch_sampled_mask]
        prediction_dict['rcnn_reg_targets'] = rcnn_reg_targets[
            batch_sampled_mask]

        # update rois_batch
        prediction_dict['rois_batch'] = rois_batch[batch_sampled_mask].view(
            rois_batch.shape[0], -1, 5)

        # return batch_sampled_mask

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
        rcnn_cls_loss = self.rcnn_cls_loss(
            rcnn_cls_scores, rcnn_cls_targets)
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
        loss_dict['rcnn_cls_targets'] = rcnn_cls_targets

        return loss_dict
