# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.autograd.functional as F

from core.model import Model
from core.anchor_generators.anchor_generator import AnchorGenerator
from core.bbox_coder import BBoxCoder
from core.samplers.hard_negative_sampler import HardNegativeSampler
from utils import box_ops
from lib.model.nms.nms_wrapper import nms
from core.losses.smooth_l1_loss import SmoothL1Loss


class RPNModel(Model):
    def build(self, model_config):
        self.anchor_scales = model_config['anchor_scales']
        self.anchor_ratios = model_config['anchor_ratios']
        self.feat_stride = model_config['feat_stride']
        self.in_channels = model_config['din']
        self.post_nms_topN = model_config['post_nms_topN']
        self.pre_nms_topN = model_config['pre_nms_topN']
        self.nms_thresh = model_config['nms_thresh']
        self.num_cls_samples = model_config['num_cls_samples']
        self.num_bbox_samples = model_config['num_bbox_samples']
        self.rpn_positive_overlap = model_config['rpn_positive_overlap']

        # define the convrelu layers processing input feature map
        self.rpn_conv = nn.Conv2d(self.in_channels, 512, 3, 1, 1, bias=True)

        self.num_anchors = len(self.anchor_scales) * len(self.anchor_ratios)

        # define bg/fg classifcation score layer
        self.nc_score_out = self.num_anchors * 2
        self.rpn_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        self.nc_bbox_out = 4 * self.num_anchors
        self.use_score = model_config['use_score']

        if self.use_score:
            bbox_feat_channels = 512 + 2
            self.nc_bbox_out /= self.num_anchors
        else:
            bbox_feat_channels = 512
        self.rpn_bbox_pred = nn.Conv2d(bbox_feat_channels, self.nc_bbox_out, 1,
                                       1, 0)

        # define proposal layer
        self.rpn_proposal = _ProposalLayer(layer_config)

        # define anchor target layer
        self.rpn_anchor_target = _AnchorTargetLayer(layer_config)

        # anchor generator
        self.anchor_generator = AnchorGenerator()

        # bbox coder
        self.bbox_coder = BBoxCoder()

        # sampler for bboxes and scores
        self.sampler = HardNegativeSampler(model_config[''])

        self.smooth_l1_loss = SmoothL1Loss()

        # self.rpn_loss_cls = 0
        # self.rpn_loss_box = 0

    def generate_proposal(self, scores, anchors, bbox_deltas):

        bbox_deltas = rpn_bbox_deltas.permute(0, 2, 3, 1).contiguous()
        # shape(N,H*W*num_anchors,4)
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)

        # apply deltas to anchors to decode
        proposals = self.bbox_coder.decode(anchors, bbox_deltas)

        # filer and clip
        proposals = box_ops.clip_boxes(proposals, im_info)

        # fg prob
        # shape(N,H*W*num_anchors)
        rpn_cls_prob = rpn_cls_prob.permute(0, 2, 1)
        scores = rpn_cls_prob[:, :, 1].view(batch_size, -1)
        # sort fg
        _, scores_order = torch.sort(fg_prob, dim=1, True)

        scores_batch = torch.zeros(batch_size, self.post_nms_topN)
        rois_batch = torch.zeros(batch_size, self.post_nms_topN, 5)

        for i in range(batch_size):
            proposal = proposals[i]
            score = scores[i]

            order_single = scores_order[i]

            # pre nms
            if self.pre_nms_topN > 0:
                order_single = order_single[:self.pre_nms_topN]

            proposal = proposal[order_single, :]
            score = score[order_single]

            # nms
            keep_idx_i = nms(
                torch.cat((proposal, scores_single), 1), self.nms_thresh)
            keep_idx_i = keep_idx_i.long().view(-1)

            # post nms
            if self.post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:self.post_nms_topN]
            proposal = proposal[keep_idx_i, :]
            score = score[keep_idx_i]

            # padding 0 at the end.
            num_proposal = keep_idx_i.numel()
            rois_batch[i, :, 0] = i
            rois_batch[i, :num_proposal, 1:] = proposal
            scores_batch[i, :num_proposal] = score
        return rois_batch, scores_batch

    def generate_anchor_target(self, anchors, gt_boxes_batch, scores_batch):
        """
        gt_boxes: shape(N,max_num_gt_boxes,5)
        anchors: shape(num_anchors,4)
        scores_batch: shape(N,num=N*W*num_anchors)
        """
        batch_size = gt_boxes.shape[0]
        num_all_samples = scores_batch.shape[1]
        labels = []

        bbox_targets_batch = torch.zeros(batch_size, self.num_bbox_samples)
        labels_batch = torch.empty(batch_size, self.num_cls_samples).fill_(-1)
        mb_bbox_mask_batch = torch.zeros(batch_size, num_all_samples)
        mb_cls_mask_batch = torch.zeros(batch_size, num_all_samples)

        # shape(num_anchors,4)
        filter_mask, filtered_anchors = box_ops.filter_window(anchors, window)
        for i in range(batch_size):
            # shape(num_anchors,num_gt)
            gt_boxes = gt_boxes_batch[i]
            scores = scores_batch[i]
            overlaps = box_ops.overlaps(filtered_anchors, gt_boxes)
            max_overlaps, argmax_overlaps = torch.max(overlaps, dim=1)

            # sample cls
            # shape(num_all_samples,)
            mb_cls_mask, mb_cls_pos_mask = self.sampler.subsample(
                filter_mask, max_overlaps, scores)
            # sample bbox
            mb_bbox_mask = self.sampler.subsample(filter_mask, max_overlaps)

            # mask
            mb_cls_mask_batch[i] = mb_cls_mask
            mb_bbox_mask_batch[i] = mb_bbox_mask

            # bbox target
            bbox_targets = self.bbox_coder.encode(anchors, gt_boxes)
            bbox_targets_batch[i] = bbox_targets[mb_bbox_mask]

            # label
            labels_batch[i][mb_cls_mask] = 0
            labels_batch[i][mb_cls_pos_mask] = 1
        return mb_bbox_mask_batch, mb_cls_mask_batch, labels_batch, bbox_targets_batch

    def forward(self, bottom_blobs):
        # rpn 3*3 conv
        # shape(N,C,H,W)
        base_feat = bottom_blobs['base_feat']
        batch_size = base_feat.shape[0]
        im_info = bottom_blobs['im_info']
        gt_boxes = bottom_blobs['gt_boxes']
        num_boxes = bottom_blobs['num_boxes']
        rpn_conv = F.relu(self.rpn_conv(base_feat))

        # rpn cls score
        # shape(N,2*num_anchors,H,W)
        rpn_cls_score = self.rpn_cls_score(rpn_conv)

        rpn_cls_score_reshape = rpn_cls_score.view(batch_size, 2, -1)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, dim=1)
        rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 1)

        # rpn bbox pred
        # shape(N,4*num_anchors,H,W)
        if self.use_score:
            rpn_bbox_preds = []
            for i in range(self.num_anchors):
                rpn_bbox_feat = torch.cat(
                    [rpn_conv, rpn_cls_score[:, ::self.num_anchors, :, :]],
                    dim=1)
                rpn_bbox_preds.append(self.RPN_bbox_pred(rpn_bbox_feat))
            rpn_bbox_deltas = torch.cat(rpn_bbox_preds, dim=1)
        else:
            # get rpn offsets to the anchor boxes
            rpn_bbox_deltas = self.RPN_bbox_pred(rpn_conv)

        # generate anchors
        anchors = self.anchor_generator.generate_anchors()

        ###############################
        # Proposal
        ###############################

        rois_batch, scores_batch = self.generate_proposal(scores, proposals)

        #############################
        # Anchor Target
        #############################
        if self.training:
            mb_bbox_mask_batch, mb_cls_mask_batch, labels_batch, bbox_targets_batch = self.generate_anchor_target(
                anchors, gt_boxes, scores_batch)

        predict_dict = {
            # some mask
            'mb_bbox_mask_batch': mb_bbox_mask_batch,
            'mb_cls_mask_batch': mb_cls_mask_batch,

            # gt
            'label_batch': label_batch,
            'bbox_targets_batch': bbox_targets_batch,

            # pred
            # shape()
            'rpn_bbox_deltas': bbox_deltas,
            # shape(N,num,2)
            'rpn_cls_scores': rpn_cls_scores,
        }

        if self.training:
            rpn_cls_loss, rpn_bbox_loss = self.loss(predict_dict)
            predict_dict['rpn_bbox_loss'] = rpn_bbox_loss
            predict_dict['rpn_cls_loss'] = rpn_cls_loss

        return predict_dict

    def loss(self, predict_dict):
        # loss for cls
        label_batch = predict_dict['label_batch']
        rpn_cls_score = predict_dict['rpn_cls_scores']
        rpn_cls_score = rpn_cls_score[mb_cls_mask]
        rpn_cls_loss = F.cross_entropy(rpn_cls_score, rpn_label)

        # loss for bbox
        # shape(N,num,4)
        bbox_targets_batch = predict_dict['bbox_targets_batch']
        rpn_bbox_deltas = predict_dict['rpn_bbox_deltas']
        rpn_bbox_deltas = rpn_bbox_deltas[mb_bbox_mask_batch]
        rpn_bbox_loss = self.smooth_l1_loss(rpn_bbox_deltas,
                                            bbox_targets_batch)
        return rpn_cls_loss, rpn_bbox_loss
