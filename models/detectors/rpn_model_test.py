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
from models import feature_extractors


@DETECTORS.register('rpn_test')
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

        proposals_order = torch.zeros(
            batch_size, self.post_nms_topN).fill_(-1).type_as(fg_probs_order)

        return proposals, proposals_order

    def init_modules(self):
        super().init_modules()
        self.feature_extractor = feature_extractors.build(
            self.feature_extractor_config)

    def init_param(self, model_config):
        super().init_param(model_config['rpn_config'])
        self.feature_extractor_config = model_config[
            'feature_extractor_config']

    def forward(self, feed_dict):

        rpn_feat_maps, rcnn_feat_maps = self.feature_extractor.first_stage_feature(
            feed_dict[constants.KEY_IMAGE])
        feed_dict['rpn_feat_maps'] = rpn_feat_maps

        rpn_feat_maps = feed_dict['rpn_feat_maps']
        batch_size = rpn_feat_maps[0].shape[0]
        im_info = feed_dict[constants.KEY_IMAGE_INFO]

        rpn_cls_scores = []
        # rpn_cls_probs = []
        rpn_bbox_preds = []

        for rpn_feat_map in rpn_feat_maps:
            # rpn conv
            rpn_conv = F.relu(self.rpn_conv(rpn_feat_map), inplace=True)

            # shape(N,num_anchors*K,H,W)
            rpn_cls_score = self.rpn_cls_score(rpn_conv)
            rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1).contiguous()
            rpn_cls_score = rpn_cls_score.view(batch_size, -1, 2)

            rpn_bbox_pred = self.rpn_bbox_pred(rpn_conv)
            rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).contiguous()
            rpn_bbox_pred = rpn_bbox_pred.view(batch_size, -1, 4)

            rpn_cls_scores.append(rpn_cls_score)
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
            'rpn_cls_scores': rpn_cls_scores,
            'anchors': anchors,

            # used for loss
            'rpn_bbox_preds': rpn_bbox_preds,
            'rpn_cls_probs': rpn_cls_probs,
        }

        if self.training:
            loss_dict = self.loss(predict_dict, feed_dict)
            return predict_dict, loss_dict
        else:
            image_info = feed_dict[constants.KEY_IMAGE_INFO]
            proposals_batch[:, :, ::
                            2] = proposals_batch[:, :, ::
                                                 2] / image_info[:,
                                                                 3].unsqueeze(
                                                                     -1
                                                                 ).unsqueeze(
                                                                     -1)
            proposals_batch[:, :, 1::
                            2] = proposals_batch[:, :, 1::
                                                 2] / image_info[:,
                                                                 2].unsqueeze(
                                                                     -1
                                                                 ).unsqueeze(
                                                                     -1)
            predict_dict[constants.KEY_BOXES_2D] = proposals_batch
            predict_dict[constants.KEY_BOXES_2D] = proposals_batch
            predict_dict[constants.KEY_CLASSES] = rpn_cls_probs
            return predict_dict

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
        rpn_gt_label = torch.ones_like(gt_labels)
        # rpn_gt_label[gt_labels==2] = 0
        auxiliary_dict[constants.KEY_CLASSES] = rpn_gt_label
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

        cls_targets = cls_target['target']
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.long().sum().clamp(min=1).float()
        rpn_cls_loss = common_loss.calc_loss(
            self.rpn_cls_loss, cls_target, normalize=False)
        rpn_reg_loss = common_loss.calc_loss(self.rpn_bbox_loss, reg_target)
        loss_dict.update({
            'rpn_cls_loss': rpn_cls_loss / num_pos,
            'rpn_reg_loss': rpn_reg_loss
        })

        return loss_dict
