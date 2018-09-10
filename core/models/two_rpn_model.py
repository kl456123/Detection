# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import Model
# from core.models.rpn_model import RPNModel
from core.models.first_rpn_model import FirstRPNModel
from core.models.focal_loss import FocalLoss
from model.roi_align.modules.roi_align import RoIAlignAvg

from core.filler import Filler
from core.target_assigner import TargetAssigner
from core.samplers.hard_negative_sampler import HardNegativeSampler
from core.models.feature_extractor_model import FeatureExtractor

from lib.model.utils.net_utils import _smooth_l1_loss
# from lib.model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer as ProposalTarget
from lib.model.rpn.proposal_target_layer_tworpn import _ProposalTargetLayer as ProposalTargetTwoRPN


class TwoRPNModel(Model):
    def clean_base_feat(self, base_feat, rois_batch, gt_boxes=None):
        """
        Args:
            base_feat: shape(N,C,H,W)
            rois: shape(N,M,5)
        Returns:
            clean_feat: shape(N,C,H,W)
        """
        base_feat = base_feat

        upsampled_feat = self.upsample(base_feat)
        rois_batch = rois_batch[:, :, 1:]
        if gt_boxes is not None:
            rois_batch = torch.cat([rois_batch, gt_boxes[:, :, :4]], dim=1)

        rois_batch = rois_batch.int()
        batch_size = rois_batch.shape[0]
        rois_per_img = rois_batch.shape[1]
        mask = torch.zeros(upsampled_feat.shape[0], upsampled_feat.shape[2],
                           upsampled_feat.shape[3])
        for i in range(batch_size):
            # copy
            rois = rois_batch[i]
            for j in range(rois_per_img):
                roi = rois[j]
                mask[i, roi[1]:roi[3], roi[0]:roi[2]] = 1

        upsampled_feat *= mask.type_as(upsampled_feat)
        clean_feat = F.upsample(
            upsampled_feat, size=base_feat.shape[-2:], mode='bilinear')

        return clean_feat

    def second_rpn_bbox_select(self, second_rpn_bbox_pred, proposals_order):
        """
        Args:
            proposals_order: shape(batch_size,rois_per_img)
            second_rpn_bbox_pred: shape(N,A*4,H,W)
        Returns:
            res_batch: shape (batch_size,rois_per_img,4)
        """
        batch_size = proposals_order.shape[0]
        second_rpn_bbox_pred = second_rpn_bbox_pred.permute(
            0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        res_batch = second_rpn_bbox_pred.new(batch_size,
                                             proposals_order.shape[1], 4)
        for i in range(batch_size):
            bbox_single = second_rpn_bbox_pred[i]
            order_single = proposals_order[i]
            res_batch[i] = bbox_single[order_single]
        return res_batch

    def second_rpn_anchors_select(self, anchors, proposals_order):
        """
        Args:
            anchors: shape(K*A,4)
            proposals_order: shape(batch_size,rois_per_img)
        """
        return anchors[0][proposals_order]

    def second_rpn_cls_select(self, second_rpn_cls_score, proposals_order):
        """
        note that nclasses =2 here
        Args:
            proposals_order: shape(batch_size,rois_per_img)
            second_rpn_cls_score: shape(N,2*A,H,W)
        Returns:
            res_batch: shape(batch_size,rois_per_img,2)
        """
        batch_size = proposals_order.shape[0]
        h, w = second_rpn_cls_score.shape[-2:]
        second_rpn_cls_score = second_rpn_cls_score.view(
            batch_size, 2, -1, h, w).permute(
                0, 3, 4, 2, 1).contiguous().view(batch_size, -1, 2)
        res_batch = second_rpn_cls_score.new(batch_size,
                                             proposals_order.shape[1], 2)
        for i in range(batch_size):
            cls_single = second_rpn_cls_score[i]
            order_single = proposals_order[i]
            res_batch[i] = cls_single[order_single]
        return res_batch

    def forward(self, feed_dict):
        # import ipdb
        # ipdb.set_trace()
        if self.training:
            gt_boxes = feed_dict['gt_boxes']
            gt_labels = feed_dict['gt_labels']
            gt_boxes = torch.cat([gt_boxes, gt_labels.unsqueeze(2).float()],
                                 dim=2)
        else:
            gt_boxes = None
        img = feed_dict['img']
        batch_size = img.shape[0]
        prediction_dict = {}

        # base model
        base_feat = self.feature_extractor.first_stage_feature(img)
        feed_dict.update({'base_feat': base_feat})

        # rpn model
        prediction_dict.update(self.rpn_model.forward(feed_dict))

        rois_batch = prediction_dict['rois_batch']
        # shape(N,K*A)
        proposals_order = prediction_dict['proposals_order']

        if self.training:
            roi_data = self.RCNN_proposal_target(
                rois_batch, gt_boxes, proposals_order, self.bbox_coder)
            rois_batch, rois_label, rois_target, rois_inside_ws, rois_outside_ws, proposals_order = roi_data

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

        cleaned_feat = self.clean_base_feat(base_feat, rois_batch, gt_boxes)
        second_rpn_conv1 = F.relu(
            self.second_rpn_conv(cleaned_feat), inplace=True)

        # cls
        # shape(N,2*A,H,W)
        second_rpn_cls_score = self.second_rpn_cls_score(second_rpn_conv1)
        # second_rpn_cls_score_reshape = second_rpn_cls_score.view(
        # second_rpn_conv1.shape[0], 2, -1)
        # second_rpn_cls_prob_reshape = F.softmax(
        # second_rpn_cls_score_reshape, dim=1)
        # second_rpn_cls_prob = second_rpn_cls_prob_reshape.view_as(
        # second_rpn_cls_score)

        # reg
        # shape(N,A*4,H,W)
        second_rpn_bbox_pred = self.second_rpn_bbox_pred(second_rpn_conv1)

        # mask select
        proposals_order = proposals_order.long()
        second_rpn_bbox_pred = self.second_rpn_bbox_select(
            second_rpn_bbox_pred, proposals_order)
        second_rpn_cls_score = self.second_rpn_cls_select(second_rpn_cls_score,
                                                          proposals_order)

        second_rpn_anchors = self.second_rpn_anchors_select(
            prediction_dict['anchors'], proposals_order)

        if self.training:
            # cls loss
            rcnn_loss_cls = F.cross_entropy(
                second_rpn_cls_score.view(-1, 2), rois_label)

            # box loss
            rcnn_loss_bbox = _smooth_l1_loss(
                second_rpn_bbox_pred.view(-1, 4), rois_target, rois_inside_ws,
                rois_outside_ws)
            second_rpn_cls_prob = 0
        else:
            second_rpn_cls_prob = F.softmax(
                second_rpn_cls_score.view(-1, 2), dim=1).view(batch_size, -1,
                                                              2)
            rcnn_loss_cls = 0
            rcnn_loss_bbox = 0

        prediction_dict.update({
            'rois_batch': rois_batch,
            'rcnn_cls_targets': rois_label,
            'rcnn_cls_probs': second_rpn_cls_prob,
            'rcnn_bbox_preds': second_rpn_bbox_pred,
            'rcnn_cls_loss': rcnn_loss_cls,
            'rcnn_bbox_loss': rcnn_loss_bbox,
            'second_rpn_anchors': second_rpn_anchors
        })

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
        self.rpn_model = FirstRPNModel(self.rpn_config)
        # self.rcnn_pooling = RoIAlignAvg(self.pooling_size, self.pooling_size,
        # 1.0 / 16.0)
        # self.l2loss = nn.MSELoss(reduce=False)

        self.rcnn_cls_pred = nn.Linear(2048, self.n_classes)
        if self.class_agnostic:
            self.rcnn_bbox_pred = nn.Linear(2048, 4)
        else:
            self.rcnn_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

        # loss module
        # if self.use_focal_loss:
        # self.rcnn_cls_loss = FocalLoss(2)
        # else:
        # self.rcnn_cls_loss = F.cross_entropy

        # self.rcnn_bbox_loss = nn.modules.SmoothL1Loss(reduce=False)

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
        # use_org = True
        # if use_org:

        self.RCNN_proposal_target = ProposalTargetTwoRPN(
            self.proposal_target_layer_config)
        # else:
        # self.RCNN_proposal_target = ProposalTarget(
        # self.proposal_target_layer_config)

        self.din = 1024
        self.num_anchors = 9
        self.nc_bbox_out = 4 * self.num_anchors
        self.nc_score_out = self.num_anchors * 2
        # self.second_rpn_conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)
        self.second_rpn_conv = self.make_second_rpn_conv()
        self.second_rpn_cls_score = nn.Conv2d(1024, self.nc_score_out, 1, 1, 0)
        self.second_rpn_bbox_pred = nn.Conv2d(1024, self.nc_bbox_out, 1, 1, 0)

        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear')

    def make_second_rpn_conv(self):
        layers = []
        layers.append(nn.Conv2d(self.din, 512, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(512))
        layers.append(nn.Conv2d(512, 512, 3, 1, 1, bias=False))
        layers.append(nn.Conv2d(512, self.din, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(self.din))
        return nn.Sequential(*layers)

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
        self.sampler = HardNegativeSampler(model_config['sampler_config'])

        # coder
        self.bbox_coder = self.target_assigner.bbox_coder

    def loss(self, prediction_dict, feed_dict):
        """
        assign proposals label and subsample from them
        Then calculate loss
        """
        loss_dict = {}
        # assign label
        rois_batch = prediction_dict['rois_batch']
        gt_boxes = feed_dict['gt_boxes']

        gt_labels = feed_dict['gt_labels']
        rcnn_cls_targets, rcnn_reg_targets, \
            rcnn_cls_weights, rcnn_reg_weights = \
            self.target_assigner.assign(rois_batch, gt_boxes, gt_labels)

        ##########################
        # subsampler
        ##########################
        # rcnn_cls_score = prediction_dict['rcnn_cls_score']
        rcnn_cls_probs = prediction_dict['rcnn_cls_probs']
        pos_indicator = rcnn_cls_targets.type(torch.cuda.ByteTensor)

        # pos prob shape(N,K,num_classes)
        # rcnn_cls_targets shape(N,K,)
        # too ulgy
        # get accord probs for each target
        targets_indx = torch.arange(
            rcnn_cls_targets.numel()).type_as(rcnn_cls_targets)
        cls_criterion = rcnn_cls_probs.view(-1, self.n_classes)[
            targets_indx, rcnn_cls_targets.view(-1)].view_as(rcnn_cls_targets)

        reg_criterion = self.target_assigner.matcher.assigned_overlaps_batch
        if self.subsample_twice:
            # shape(N,K)
            # subsample from all
            cls_batch_sampled_mask = self.sampler.subsample_batch(
                self.rcnn_batch_size, pos_indicator, criterion=cls_criterion)
            rcnn_cls_weights *= cls_batch_sampled_mask

            # subsample from pos indicator
            reg_batch_sampled_mask = self.sampler.subsample_batch(
                self.rcnn_batch_size,
                pos_indicator,
                # subsample from iou
                criterion=reg_criterion,
                indicator=pos_indicator)
            rcnn_reg_weights *= reg_batch_sampled_mask
        else:
            # subsample from all
            batch_sampled_mask = self.sampler.subsample_batch(
                self.rcnn_batch_size, pos_indicator, criterion=cls_criterion)
            batch_sampled_mask = batch_sampled_mask.type_as(rcnn_cls_weights)
            rcnn_cls_weights = rcnn_cls_weights * batch_sampled_mask
            rcnn_reg_weights = rcnn_reg_weights * batch_sampled_mask

        # subsample
        # self.sampler.subsample()

        # submodule loss
        loss_dict.update(self.rpn_model.loss(prediction_dict, feed_dict))

        # classification loss

        rcnn_cls_scores = prediction_dict['rcnn_cls_scores']
        rcnn_cls_loss = self.rcnn_cls_loss(rcnn_cls_scores, rcnn_cls_targets)
        rcnn_cls_loss *= rcnn_cls_weights.detach()

        # bounding box regression L1 loss
        rcnn_bbox_preds = prediction_dict['rcnn_bbox_preds']
        rcnn_bbox_loss = self.rcnn_bbox_loss(rcnn_bbox_preds,
                                             rcnn_reg_targets).sum(dim=-1)
        rcnn_bbox_loss *= rcnn_reg_weights.detach()

        # loss weights has no gradients
        loss_dict['rcnn_cls_loss'] = rcnn_cls_loss
        loss_dict['rcnn_bbox_loss'] = rcnn_bbox_loss

        # add rcnn_cls_targets to get the statics of rpn
        loss_dict['rcnn_cls_targets'] = rcnn_cls_targets

        return loss_dict
