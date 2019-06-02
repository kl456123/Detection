# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.nn.functional as F

from core.model import Model
from core.filler import Filler
from models.losses.focal_loss import FocalLoss

from utils import box_ops
from lib.model.roi_layers import nms
import functools
from utils.registry import DETECTORS
from target_generators.target_generator import TargetGenerator
# import samplers
import anchor_generators
from models.losses import common_loss
from core import constants
import bbox_coders


@DETECTORS.register('rpn')
class RPNModel(Model):
    def init_param(self, model_config):
        self.in_channels = model_config['din']
        self.post_nms_topN = model_config['post_nms_topN']
        self.pre_nms_topN = model_config['pre_nms_topN']
        self.nms_thresh = model_config['nms_thresh']
        self.use_focal_loss = model_config['use_focal_loss']

        # anchor generator
        self.anchor_generator = anchor_generators.build(
            model_config['anchor_generator_config'])
        self.num_anchors = self.anchor_generator.num_anchors
        self.nc_bbox_out = 4 * self.num_anchors
        self.nc_score_out = self.num_anchors * 2

        self.target_generators = TargetGenerator(
            model_config['target_generator_config'])

    def init_weights(self):
        self.truncated = False

        Filler.normal_init(self.rpn_conv, 0, 0.01, self.truncated)
        Filler.normal_init(self.rpn_cls_score, 0, 0.01, self.truncated)
        Filler.normal_init(self.rpn_bbox_pred, 0, 0.01, self.truncated)

    def init_modules(self):
        # define the convrelu layers processing input feature map
        self.rpn_conv = nn.Conv2d(self.in_channels, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        self.rpn_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer

        bbox_feat_channels = 512
        self.rpn_bbox_pred = nn.Conv2d(bbox_feat_channels, self.nc_bbox_out, 1,
                                       1, 0)

        # bbox
        self.rpn_bbox_loss = nn.SmoothL1Loss(reduction='none')

        # cls
        if self.use_focal_loss:
            self.rpn_cls_loss = FocalLoss(2, gamma=2, alpha=0.25)
        else:
            self.rpn_cls_loss = nn.CrossEntropyLoss(reduction='none')

    def generate_proposal(self, rpn_cls_probs, anchors, rpn_bbox_preds,
                          im_info):
        # TODO create a new Function
        """
        Args:
        rpn_cls_probs: FloatTensor,shape(N,2*num_anchors,H,W)
        rpn_bbox_preds: FloatTensor,shape(N,num_anchors*4,H,W)
        anchors: FloatTensor,shape(N,4,H,W)

        Returns:
        proposals_batch: FloatTensor, shape(N,post_nms_topN,4)
        fg_probs_batch: FloatTensor, shape(N,post_nms_topN)
        """
        # assert len(
        # rpn_bbox_preds) == 1, 'just one feature maps is supported now'
        # rpn_bbox_preds = rpn_bbox_preds[0]
        # do not backward
        rpn_cls_probs = rpn_cls_probs.detach()
        rpn_bbox_preds = rpn_bbox_preds.detach()

        batch_size = rpn_bbox_preds.shape[0]
        rpn_bbox_preds = rpn_bbox_preds.permute(0, 2, 3, 1).contiguous()
        # shape(N,H*W*num_anchors,4)
        rpn_bbox_preds = rpn_bbox_preds.view(batch_size, -1, 4)

        coders = bbox_coders.build({'type': constants.KEY_BOXES_2D})
        proposals = coders.decode_batch(rpn_bbox_preds, anchors)

        # filer and clip
        proposals = box_ops.clip_boxes(proposals, im_info)

        # fg prob
        fg_probs = rpn_cls_probs[:, self.num_anchors:, :, :]
        fg_probs = fg_probs.permute(0, 2, 3, 1).contiguous().view(
            batch_size, -1)

        # sort fg
        _, fg_probs_order = torch.sort(fg_probs, dim=1, descending=True)

        # fg_probs_batch = torch.zeros(batch_size,
        # self.post_nms_topN).type_as(rpn_cls_probs)
        proposals_batch = torch.zeros(batch_size, self.post_nms_topN,
                                      4).type_as(rpn_bbox_preds)
        proposals_order = torch.zeros(
            batch_size, self.post_nms_topN).fill_(-1).type_as(fg_probs_order)

        for i in range(batch_size):
            proposals_single = proposals[i]
            fg_probs_single = fg_probs[i]
            fg_order_single = fg_probs_order[i]
            # pre nms
            if self.pre_nms_topN > 0:
                fg_order_single = fg_order_single[:self.pre_nms_topN]
            proposals_single = proposals_single[fg_order_single]
            fg_probs_single = fg_probs_single[fg_order_single]

            # nms
            keep_idx_i = nms(proposals_single, fg_probs_single,
                             self.nms_thresh)
            keep_idx_i = keep_idx_i.long().view(-1)

            # post nms
            if self.post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:self.post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            fg_probs_single = fg_probs_single[keep_idx_i]
            fg_order_single = fg_order_single[keep_idx_i]

            # padding 0 at the end.
            num_proposal = keep_idx_i.numel()
            proposals_batch[i, :num_proposal, :] = proposals_single
            # fg_probs_batch[i, :num_proposal] = fg_probs_single
            proposals_order[i, :num_proposal] = fg_order_single
        return proposals_batch, proposals_order

    def forward(self, bottom_blobs):
        base_feat = bottom_blobs['base_feat']
        batch_size = base_feat.shape[0]
        im_info = bottom_blobs[constants.KEY_IMAGE_INFO]

        # rpn conv
        rpn_conv = F.relu(self.rpn_conv(base_feat), inplace=True)

        # rpn cls score
        # shape(N,2*num_anchors,H,W)
        rpn_cls_scores = self.rpn_cls_score(rpn_conv)

        # rpn cls prob shape(N,2*num_anchors,H,W)
        rpn_cls_score_reshape = rpn_cls_scores.view(batch_size, 2, -1)
        rpn_cls_probs = F.softmax(rpn_cls_score_reshape, dim=1)
        rpn_cls_probs = rpn_cls_probs.view_as(rpn_cls_scores)
        # import ipdb
        # ipdb.set_trace()

        # rpn bbox pred
        # shape(N,4*num_anchors,H,W)
        rpn_bbox_preds = self.rpn_bbox_pred(rpn_conv)

        # generate anchors
        feature_map_list = [base_feat.size()[-2:]]
        anchors = self.anchor_generator.generate(feature_map_list,
                                                 im_info[0][:-1])

        anchors = anchors.unsqueeze(0).repeat(batch_size, 1, 1)

        ###############################
        # Proposal
        ###############################
        # note that proposals_order is used for track transform of propsoals
        proposals_batch, proposals_order = self.generate_proposal(
            rpn_cls_probs, anchors, rpn_bbox_preds, im_info)
        #  batch_idx = torch.arange(batch_size).view(batch_size, 1).expand(
        #  -1, proposals_batch.shape[1]).type_as(proposals_batch)
        #  rois_batch = torch.cat((batch_idx.unsqueeze(-1), proposals_batch),
        #  dim=2)

        if self.training:
            label_boxes_2d = bottom_blobs[constants.KEY_LABEL_BOXES_2D]
            proposals_batch = self.append_gt(proposals_batch, label_boxes_2d)

        rpn_cls_scores = rpn_cls_scores.view(batch_size, 2, -1,
                                             rpn_cls_scores.shape[2],
                                             rpn_cls_scores.shape[3])
        rpn_cls_scores = rpn_cls_scores.permute(0, 3, 4, 2,
                                                1).contiguous().view(
                                                    batch_size, -1, 2)

        # postprocess
        rpn_cls_probs = rpn_cls_probs.view(
            batch_size, 2, -1, rpn_cls_probs.shape[2], rpn_cls_probs.shape[3])
        rpn_cls_probs = rpn_cls_probs.permute(0, 3, 4, 2, 1).contiguous().view(
            batch_size, -1, 2)

        rpn_bbox_preds = rpn_bbox_preds.permute(0, 2, 3, 1).contiguous()
        # shape(N,H*W*num_anchors,4)
        rpn_bbox_preds = rpn_bbox_preds.view(batch_size, -1, 4)

        predict_dict = {
            'proposals': proposals_batch,
            'rpn_cls_scores': rpn_cls_scores,
            #  'rois_batch': rois_batch,
            'anchors': anchors,

            # used for loss
            'rpn_bbox_preds': rpn_bbox_preds,
            'rpn_cls_probs': rpn_cls_probs,
            'proposals_order': proposals_order,
        }

        return predict_dict

    def append_gt(self, proposals_batch, label_boxes_2d):
        """
        Args:
            proposals_batch: shape(N, M, 4)
            label_boxes_2d: shape(N, m, 4)
            num_instances: shape(N,) valid num of bboxes in each image
        Returns:
            proposals_batch: shape(N, M+m, 4)
        """
        return torch.cat([proposals_batch, label_boxes_2d], dim=1)

    def loss(self, prediction_dict, feed_dict):
        # loss for cls
        loss_dict = {}
        anchors = prediction_dict['anchors']
        anchors_dict = {}
        anchors_dict[constants.KEY_PRIMARY] = anchors
        anchors_dict[constants.KEY_BOXES_2D] = prediction_dict[
            'rpn_bbox_preds']
        anchors_dict[constants.KEY_CLASSES] = prediction_dict['rpn_cls_scores']

        gt_dict = {}
        gt_dict[constants.KEY_PRIMARY] = feed_dict[
            constants.KEY_LABEL_BOXES_2D]
        gt_dict[constants.KEY_CLASSES] = None
        gt_dict[constants.KEY_BOXES_2D] = None

        auxiliary_dict = {}
        auxiliary_dict[constants.KEY_BOXES_2D] = feed_dict[
            constants.KEY_LABEL_BOXES_2D]
        gt_labels = feed_dict[constants.KEY_LABEL_CLASSES]
        auxiliary_dict[constants.KEY_CLASSES] = torch.ones_like(gt_labels)
        auxiliary_dict[constants.KEY_NUM_INSTANCES] = feed_dict[
            constants.KEY_NUM_INSTANCES]
        auxiliary_dict[constants.KEY_PROPOSALS] = anchors

        # import ipdb
        # ipdb.set_trace()
        _, targets, _ = self.target_generators.generate_targets(
            anchors_dict, gt_dict, auxiliary_dict, subsample=False)

        cls_target = targets[constants.KEY_CLASSES]
        reg_target = targets[constants.KEY_BOXES_2D]

        # loss

        if self.use_focal_loss:
            # when using focal loss, dont normalize it by all samples
            cls_targets = cls_target['target']
            pos = cls_targets > 0  # [N,#anchors]
            num_pos = pos.long().sum().clamp(min=1).float()
            rpn_cls_loss = common_loss.calc_loss(
                self.rpn_cls_loss, cls_target, normalize=False) / num_pos
        else:
            rpn_cls_loss = common_loss.calc_loss(self.rpn_cls_loss, cls_target)
        rpn_reg_loss = common_loss.calc_loss(self.rpn_bbox_loss, reg_target)
        loss_dict.update({
            'rpn_cls_loss': rpn_cls_loss,
            'rpn_reg_loss': rpn_reg_loss
        })

        return loss_dict
