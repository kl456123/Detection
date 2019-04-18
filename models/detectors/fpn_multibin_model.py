# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.nn.functional as F

from lib.model.roi_layers import ROIAlign

from models.detectors.fpn_mono_3d_model import FPNMono3D
from models.losses.multibin_loss import MultiBinLoss
from core.filler import Filler
from core import constants

from models.losses import common_loss
from models.losses.focal_loss import FocalLoss
from models.losses.orientation_loss import OrientationLoss

from utils.registry import DETECTORS
from utils import box_ops

from target_generators.target_generator import TargetGenerator
from models import feature_extractors
from models import detectors

from utils import batch_ops
import bbox_coders
from core.utils.analyzer import Analyzer


@DETECTORS.register('fpn_multibin_mono_3d')
class FPNMultiBinMono3D(FPNMono3D):
    def forward(self, feed_dict):
        # import ipdb
        # ipdb.set_trace()
        im_info = feed_dict[constants.KEY_IMAGE_INFO]

        prediction_dict = {}

        # base model
        rpn_feat_maps, rcnn_feat_maps = self.feature_extractor.first_stage_feature(
            feed_dict[constants.KEY_IMAGE])
        feed_dict.update({'rpn_feat_maps': rpn_feat_maps})

        # rpn model
        prediction_dict.update(self.rpn_model.forward(feed_dict))
        proposals = prediction_dict['proposals']
        multi_stage_loss_units = []
        multi_stage_stats = []
        for i in range(self.num_stages):

            if self.training:
                # proposals_dict
                proposals_dict = {}
                proposals_dict[constants.KEY_PRIMARY] = proposals

                # gt_dict
                gt_dict = {}
                gt_dict[constants.KEY_PRIMARY] = feed_dict[constants.
                                                           KEY_LABEL_BOXES_2D]
                gt_dict[constants.KEY_CLASSES] = None
                gt_dict[constants.KEY_BOXES_2D] = None
                gt_dict[constants.KEY_ORIENTS_V3] = None
                gt_dict[constants.KEY_DIMS] = None

                # auxiliary_dict(used for encoding)
                auxiliary_dict = {}
                auxiliary_dict[constants.KEY_STEREO_CALIB_P2] = feed_dict[
                    constants.KEY_STEREO_CALIB_P2]
                auxiliary_dict[constants.KEY_BOXES_2D] = feed_dict[
                    constants.KEY_LABEL_BOXES_2D]
                auxiliary_dict[constants.KEY_CLASSES] = feed_dict[
                    constants.KEY_LABEL_CLASSES]
                auxiliary_dict[constants.KEY_BOXES_3D] = feed_dict[
                    constants.KEY_LABEL_BOXES_3D]
                auxiliary_dict[constants.KEY_NUM_INSTANCES] = feed_dict[
                    constants.KEY_NUM_INSTANCES]
                auxiliary_dict[constants.KEY_PROPOSALS] = proposals
                auxiliary_dict[constants.KEY_MEAN_DIMS] = feed_dict[
                    constants.KEY_MEAN_DIMS]

                proposals_dict, loss_units, stats = self.target_generators[
                    i].generate_targets(proposals_dict, gt_dict,
                                        auxiliary_dict)

                # note here base_feat (N,C,H,W),rois_batch (N,num_proposals,5)
                proposals = proposals_dict[constants.KEY_PRIMARY]
            rois = box_ops.box2rois(proposals)
            pooled_feat = self.pyramid_rcnn_pooling(rcnn_feat_maps,
                                                    rois.view(-1, 5),
                                                    im_info[0][:2])

            # shape(N,C,1,1)
            pooled_feat = self.feature_extractor.second_stage_feature(
                pooled_feat)
            pooled_feat = pooled_feat.mean(3).mean(2)

            rcnn_bbox_preds = self.rcnn_bbox_preds[i](pooled_feat)
            rcnn_cls_scores = self.rcnn_cls_preds[i](pooled_feat)
            rcnn_orient_preds = self.rcnn_orient_preds[i](pooled_feat)
            rcnn_dim_preds = self.rcnn_dim_preds[i](pooled_feat)

            rcnn_cls_probs = F.softmax(rcnn_cls_scores, dim=1)

            batch_size = rois.shape[0]
            rcnn_cls_scores = rcnn_cls_scores.view(batch_size, -1,
                                                   self.n_classes)
            rcnn_cls_probs = rcnn_cls_probs.view(batch_size, -1,
                                                 self.n_classes)
            if not self.class_agnostic:
                # import ipdb
                # ipdb.set_trace()
                if self.training:
                    rcnn_bbox_preds = self.squeeze_bbox_preds(
                        rcnn_bbox_preds,
                        loss_units[constants.KEY_CLASSES]['target'].view(-1))
                else:
                    rcnn_bbox_preds = self.squeeze_bbox_preds(
                        rcnn_bbox_preds,
                        rcnn_cls_probs.argmax(dim=-1).view(-1))

            rcnn_bbox_preds = rcnn_bbox_preds.view(batch_size, -1, 4)
            rcnn_orient_preds = rcnn_orient_preds.view(batch_size, -1,
                                                       4 * self.num_bins)
            rcnn_dim_preds = rcnn_dim_preds.view(batch_size, -1, 3)

            if self.training:
                loss_units[constants.KEY_CLASSES]['pred'] = rcnn_cls_scores
                loss_units[constants.KEY_BOXES_2D]['pred'] = rcnn_bbox_preds
                loss_units[constants.KEY_ORIENTS_V3]['pred'] = rcnn_orient_preds
                loss_units[constants.KEY_DIMS]['pred'] = rcnn_dim_preds
                # import ipdb
                # ipdb.set_trace()
                multi_stage_loss_units.append([
                    loss_units[constants.KEY_CLASSES],
                    loss_units[constants.KEY_BOXES_2D],
                    loss_units[constants.KEY_ORIENTS_V3],
                    loss_units[constants.KEY_DIMS]
                ])
                multi_stage_stats.append(stats)

            # decode for next stage
            coder = bbox_coders.build({'type': constants.KEY_BOXES_2D})
            #  rpn_proposals = proposals
            proposals = coder.decode_batch(rcnn_bbox_preds, proposals).detach()
            coder = bbox_coders.build({'type': constants.KEY_DIMS})
            rcnn_dim_preds = coder.decode_batch(
                rcnn_dim_preds, feed_dict[constants.KEY_MEAN_DIMS],
                rcnn_cls_probs).detach()
            coder = bbox_coders.build({'type': constants.KEY_ORIENTS_V3})
            # use rpn proposals to decode
            rcnn_orient_preds = coder.decode_batch(
                rcnn_orient_preds, self.rcnn_orient_loss.bin_centers,
                proposals, feed_dict[constants.KEY_STEREO_CALIB_P2]).detach()

        if self.training:
            prediction_dict[constants.KEY_TARGETS] = multi_stage_loss_units
            prediction_dict[constants.KEY_STATS] = multi_stage_stats
        else:
            prediction_dict[constants.KEY_CLASSES] = rcnn_cls_probs
            prediction_dict[constants.KEY_ORIENTS_V3] = rcnn_orient_preds

            image_info = feed_dict[constants.KEY_IMAGE_INFO]
            proposals[:, :, ::2] = proposals[:, :, ::
                                             2] / image_info[:, 3].unsqueeze(
                                                 -1).unsqueeze(-1)
            proposals[:, :, 1::2] = proposals[:, :, 1::
                                              2] / image_info[:, 2].unsqueeze(
                                                  -1).unsqueeze(-1)
            prediction_dict[constants.KEY_BOXES_2D] = proposals
            prediction_dict[constants.KEY_DIMS] = rcnn_dim_preds
            prediction_dict[constants.KEY_ORIENTS_V3] = rcnn_orient_preds

        return prediction_dict

    def init_param(self, model_config):
        super().init_param(model_config)
        self.num_bins = model_config['num_bins']

    def init_modules(self):
        super().init_modules()
        self.rcnn_orient_preds = nn.ModuleList([
            nn.Linear(1024, 4 * self.num_bins) for _ in range(self.num_stages)
        ])

        self.rcnn_dim_preds = nn.ModuleList(
            [nn.Linear(1024, 3) for _ in range(self.num_stages)])

        self.rcnn_orient_loss = MultiBinLoss(num_bins=self.num_bins)

    #  def loss(self, prediction_dict, feed_dict):
    #  """
    #  assign proposals label and subsample from them
    #  Then calculate loss
    #  """
    #  loss_dict = super().loss(prediction_dict, feed_dict)
    #  targets = prediction_dict[constants.KEY_TARGETS]
    #  rcnn_orient_loss = 0
    #  rcnn_dim_loss = 0

    #  for stage_ind in range(self.num_stages):
    #  orient_target = targets[stage_ind][2]
    #  rcnn_orient_loss = rcnn_orient_loss + common_loss.calc_loss(
    #  self.rcnn_orient_loss, orient_target, True)

    #  dim_target = targets[stage_ind][3]
    #  rcnn_dim_loss = rcnn_dim_loss + common_loss.calc_loss(
    #  self.rcnn_bbox_loss, dim_target, True)

    #  loss_dict.update({
    #  'rcnn_orient_loss': rcnn_orient_loss,
    #  'rcnn_dim_loss': rcnn_dim_loss
    #  })

    #  return loss_dict
