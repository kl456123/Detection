# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.nn.functional as F

from core.model import Model
from models.detectors.rpn_model import RPNModel as _RPNModel
from core.filler import Filler
from models.losses.focal_loss import FocalLoss

from utils import box_ops
from lib.model.roi_layers import nms
from utils.registry import DETECTORS
from target_generators.target_generator import TargetGenerator
# import samplers
from models.losses import common_loss
from core import constants
import bbox_coders


@DETECTORS.register('fpn_rpn')
class RPNModel(_RPNModel):
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

        coders = bbox_coders.build(
            self.target_generators.target_generator_config['coder_config'])
        proposals = coders.decode_batch(rpn_bbox_preds, anchors)

        # filer and clip
        proposals = box_ops.clip_boxes(proposals, im_info)

        # fg prob
        fg_probs = rpn_cls_probs[:, :, 1]

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
        rpn_feat_maps = bottom_blobs['rpn_feat_maps']
        batch_size = rpn_feat_maps[0].shape[0]
        im_info = bottom_blobs[constants.KEY_IMAGE_INFO]

        rpn_cls_scores = []
        # rpn_cls_probs = []
        rpn_bbox_preds = []

        for rpn_feat_map in rpn_feat_maps:
            # rpn conv
            rpn_conv = F.relu(self.rpn_conv(rpn_feat_map), inplace=True)

            # rpn cls score
            # shape(N,2*num_anchors,H,W)
            rpn_cls_score = self.rpn_cls_score(rpn_conv)
            rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1).contiguous()
            rpn_cls_score = rpn_cls_score.view(batch_size, -1, 2)

            # rpn cls prob shape(N,2*num_anchors,H,W)
            # rpn_cls_score_reshape = rpn_cls_score.view(batch_size, 2, -1)
            # rpn_cls_prob = F.softmax(rpn_cls_score_reshape, dim=1)
            # rpn_cls_prob = rpn_cls_prob.view_as(rpn_cls_score)

            # rpn_cls_prob = rpn_cls_prob.view(batch_size, 2, -1,
            # rpn_cls_prob.shape[2],
            # rpn_cls_prob.shape[3])
            # rpn_cls_prob = rpn_cls_prob.permute(
            # 0, 3, 4, 2, 1).contiguous().view(batch_size, -1, 2)

            # rpn_cls_score = rpn_cls_score.view(batch_size, 2, -1,
            # rpn_cls_score.shape[2],
            # rpn_cls_score.shape[3])
            # rpn_cls_score = rpn_cls_score.permute(
            # 0, 3, 4, 2, 1).contiguous().view(batch_size, -1, 2)
            rpn_bbox_pred = self.rpn_bbox_pred(rpn_conv)
            # rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).contiguous()
            rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).contiguous()
            # shape(N,H*W*num_anchors,4)
            rpn_bbox_pred = rpn_bbox_pred.view(batch_size, -1, 4)

            # rpn_cls_probs.append(rpn_cls_prob)
            rpn_cls_scores.append(rpn_cls_score)
            # get rpn offsets to the anchor boxes
            rpn_bbox_preds.append(rpn_bbox_pred)

        rpn_cls_scores = torch.cat(rpn_cls_scores, dim=1)
        # rpn_cls_probs = torch.cat(rpn_cls_probs, dim=1)
        rpn_bbox_preds = torch.cat(rpn_bbox_preds, dim=1)
        rpn_cls_probs = F.softmax(rpn_cls_scores, dim=-1)

        # generate pyramid anchors
        feature_map_list = [
            base_feat.shape[-2:] for base_feat in rpn_feat_maps
        ]
        anchors = self.anchor_generator.generate_pyramid(
            feature_map_list, im_info[0][:-1])
        anchors = anchors.unsqueeze(0).repeat(batch_size, 1, 1)

        ###############################
        # Proposal
        ###############################
        # note that proposals_order is used for track transform of propsoals
        proposals_batch, proposals_order = self.generate_proposal(
            rpn_cls_probs, anchors, rpn_bbox_preds, im_info)

        # if self.training:
        # label_boxes_2d = bottom_blobs[constants.KEY_LABEL_BOXES_2D]
        # proposals_batch = self.append_gt(proposals_batch, label_boxes_2d)

        # postprocess

        predict_dict = {
            'proposals': proposals_batch,
            'rpn_cls_scores': rpn_cls_scores,
            'anchors': anchors,

            # used for loss
            'rpn_bbox_preds': rpn_bbox_preds,
            'rpn_cls_probs': rpn_cls_probs,
        }

        return predict_dict
