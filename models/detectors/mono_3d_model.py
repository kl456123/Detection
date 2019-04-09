# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.nn.functional as F

from lib.model.roi_align.modules.roi_align import RoIAlignAvg

from core.model import Model
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


@DETECTORS.register('mono_3d')
class Mono3D(Model):
    def forward(self, feed_dict):
        # import ipdb
        # ipdb.set_trace()

        prediction_dict = {}

        # base model
        base_feat = self.feature_extractor.first_stage_feature(
            feed_dict[constants.KEY_IMAGE])
        feed_dict.update({'base_feat': base_feat})
        self.add_feat('base_feat', base_feat)

        # rpn model
        prediction_dict.update(self.rpn_model.forward(feed_dict))
        proposals = prediction_dict['proposals']
        multi_stage_loss_units = []
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
                gt_dict[constants.KEY_ORIENTS] = None
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

                proposals_dict, loss_units = self.target_generators[
                    i].generate_targets(proposals_dict, gt_dict,
                                        auxiliary_dict)

                # note here base_feat (N,C,H,W),rois_batch (N,num_proposals,5)
                proposals = proposals_dict[constants.KEY_PRIMARY]
            rois = box_ops.box2rois(proposals)
            pooled_feat = self.rcnn_pooling(base_feat, rois.view(-1, 5))

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
            rcnn_orient_preds = rcnn_orient_preds.view(batch_size, -1, 4)
            rcnn_dim_preds = rcnn_dim_preds.view(batch_size, -1, 3)

            if self.training:
                loss_units[constants.KEY_CLASSES]['pred'] = rcnn_cls_scores
                loss_units[constants.KEY_BOXES_2D]['pred'] = rcnn_bbox_preds
                loss_units[constants.KEY_ORIENTS]['pred'] = rcnn_orient_preds
                loss_units[constants.KEY_DIMS]['pred'] = rcnn_dim_preds
                # import ipdb
                # ipdb.set_trace()
                multi_stage_loss_units.extend([
                    loss_units[constants.KEY_CLASSES],
                    loss_units[constants.KEY_BOXES_2D],
                    loss_units[constants.KEY_ORIENTS],
                    loss_units[constants.KEY_DIMS]
                ])

            # decode for next stage
            coder = bbox_coders.build({'type': constants.KEY_BOXES_2D})
            rpn_proposals = proposals
            proposals = coder.decode_batch(rcnn_bbox_preds, proposals).detach()
            coder = bbox_coders.build({'type': constants.KEY_DIMS})
            rcnn_dim_preds = coder.decode_batch(
                rcnn_dim_preds, feed_dict[constants.KEY_MEAN_DIMS],
                rcnn_cls_probs).detach()
            coder = bbox_coders.build({'type': constants.KEY_ORIENTS})
            # use rpn proposals to decode
            rcnn_orient_preds = coder.decode_batch(
                rcnn_orient_preds, rpn_proposals, proposals,
                feed_dict[constants.KEY_STEREO_CALIB_P2]).detach()

        if self.training:
            prediction_dict[constants.KEY_TARGETS] = multi_stage_loss_units
        else:
            prediction_dict[constants.KEY_CLASSES] = rcnn_cls_probs
            prediction_dict[constants.KEY_ORIENTS] = rcnn_orient_preds

            image_info = feed_dict[constants.KEY_IMAGE_INFO]
            proposals[:, :, ::2] = proposals[:, :, ::
                                             2] / image_info[:, 3].unsqueeze(
                                                 -1).unsqueeze(-1)
            proposals[:, :, 1::2] = proposals[:, :, 1::
                                              2] / image_info[:, 2].unsqueeze(
                                                  -1).unsqueeze(-1)
            prediction_dict[constants.KEY_BOXES_2D] = proposals
            prediction_dict[constants.KEY_DIMS] = rcnn_dim_preds
            prediction_dict[constants.KEY_ORIENTS] = rcnn_orient_preds

        return prediction_dict

    def squeeze_bbox_preds(self, rcnn_bbox_preds, rcnn_cls_targets, out_c=4):
        """
        squeeze rcnn_bbox_preds from shape (N, 4 * num_classes) to shape (N, 4)
        Args:
            rcnn_bbox_preds: shape(N, num_classes, 4)
            rcnn_cls_targets: shape(N, 1)
        """
        rcnn_bbox_preds = rcnn_bbox_preds.view(-1, self.n_classes, out_c)
        batch_size = rcnn_bbox_preds.shape[0]
        offset = torch.arange(0, batch_size) * rcnn_bbox_preds.size(1)
        rcnn_cls_targets = rcnn_cls_targets + offset.type_as(rcnn_cls_targets)
        rcnn_bbox_preds = rcnn_bbox_preds.contiguous().view(
            -1, out_c)[rcnn_cls_targets]
        return rcnn_bbox_preds

    def init_weights(self):
        # submodule init weights
        self.feature_extractor.init_weights()
        self.rpn_model.init_weights()

        # Filler.normal_init(self.rcnn_cls_preds, 0, 0.01, self.truncated)
        # Filler.normal_init(self.rcnn_bbox_preds, 0, 0.001, self.truncated)

    def init_modules(self):
        self.feature_extractor = feature_extractors.build(
            self.feature_extractor_config)
        self.rpn_model = detectors.build(self.rpn_config)
        if self.pooling_mode == 'align':
            self.rcnn_pooling = RoIAlignAvg(self.pooling_size,
                                            self.pooling_size, 1.0 / 16.0)
        # note that roi extractor is shared but heads not
        self.rcnn_cls_preds = nn.ModuleList(
            [nn.Linear(2048, self.n_classes) for _ in range(self.num_stages)])
        in_channels = 2048
        if self.class_agnostic:
            rcnn_bbox_pred = nn.Linear(in_channels, 4)
        else:
            rcnn_bbox_pred = nn.Linear(in_channels, 4 * self.n_classes)
        self.rcnn_bbox_preds = nn.ModuleList(
            [rcnn_bbox_pred for _ in range(self.num_stages)])
        self.rcnn_orient_preds = nn.ModuleList(
            [nn.Linear(2048, 4) for _ in range(self.num_stages)])

        self.rcnn_dim_preds = nn.ModuleList(
            [nn.Linear(2048, 3) for _ in range(self.num_stages)])

        # loss module
        # if self.use_focal_loss:
        # self.rcnn_cls_loss = FocalLoss(self.n_classes)
        # else:
        self.rcnn_cls_loss = nn.CrossEntropyLoss(reduce=False)

        self.rcnn_bbox_loss = nn.modules.SmoothL1Loss(reduce=False)

        self.rcnn_orient_loss = OrientationLoss()

    def init_param(self, model_config):
        classes = model_config['classes']
        self.classes = classes
        self.n_classes = len(classes) + 1
        self.class_agnostic = model_config['class_agnostic']
        self.pooling_size = model_config['pooling_size']
        self.pooling_mode = model_config['pooling_mode']
        self.truncated = model_config['truncated']
        self.use_focal_loss = model_config['use_focal_loss']

        # some submodule config
        self.feature_extractor_config = model_config['feature_extractor_config']
        self.rpn_config = model_config['rpn_config']

        self.num_stages = len(model_config['target_generator_config'])
        self.target_generators = [
            TargetGenerator(model_config['target_generator_config'][i])
            for i in range(self.num_stages)
        ]

    def loss(self, prediction_dict, feed_dict):
        """
        assign proposals label and subsample from them
        Then calculate loss
        """
        loss_dict = {}

        # import ipdb
        # ipdb.set_trace()
        # submodule loss
        loss_dict.update(self.rpn_model.loss(prediction_dict, feed_dict))

        targets = prediction_dict[constants.KEY_TARGETS]

        # cls_target = targets[0]
        # reg_target = targets[3]

        rcnn_cls_loss = 0
        rcnn_reg_loss = 0
        rcnn_orient_loss = 0
        rcnn_dim_loss = 0
        for cls_target in targets[::4]:
            rcnn_cls_loss = rcnn_cls_loss + common_loss.calc_loss(
                self.rcnn_cls_loss, cls_target)
        for reg_target in targets[1::4]:
            rcnn_reg_loss = rcnn_reg_loss + common_loss.calc_loss(
                self.rcnn_bbox_loss, reg_target)
        for orient_target in targets[2::4]:
            rcnn_orient_loss = rcnn_orient_loss + common_loss.calc_loss(
                self.rcnn_orient_loss, orient_target)
        for dim_target in targets[3::4]:
            rcnn_dim_loss = rcnn_dim_loss + common_loss.calc_loss(
                self.rcnn_bbox_loss, dim_target)

        loss_dict.update({
            'rcnn_cls_loss': rcnn_cls_loss,
            'rcnn_reg_loss': rcnn_reg_loss,
            'rcnn_orient_loss': rcnn_orient_loss,
            'rcnn_dim_loss': rcnn_dim_loss
        })

        return loss_dict
