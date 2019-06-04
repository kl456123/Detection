# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.nn.functional as F
import functools

from lib.model.roi_layers import ROIAlign
from lib.model.roi_layers import AdaptiveROIAlign

from models.detectors.fpn_faster_rcnn_model import FPNFasterRCNN
from core.filler import Filler
from core import constants

from models.losses import common_loss
from models.losses.focal_loss import FocalLoss
from models.losses.orientation_loss import OrientationLoss
from models.losses.corners_loss import CornersLoss

from utils.registry import DETECTORS
from utils import box_ops

from target_generators.target_generator import TargetGenerator
from models import feature_extractors
from models import detectors

from utils import batch_ops
import bbox_coders
from core.utils.analyzer import Analyzer
import copy


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    conv = conv3x3(in_planes, out_planes, stride=1)
    bn = nn.BatchNorm2d(out_planes)
    relu = nn.ReLU()
    return nn.Sequential(*[conv, bn, relu])


class KeyPointPredictor(nn.Module):
    def __init__(self, inplane, output=4):
        super().__init__()

        layers = []
        for i in range(8):
            layers.append(conv3x3_bn_relu(inplane, 512))
            inplane = 512

        # upsample
        deconv = nn.ConvTranspose2d(512, 512, 2, 2, 0)
        bn = nn.BatchNorm2d(512)
        relu = nn.ReLU()
        upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        conv = nn.Conv2d(512, output, 1, 1, 0)
        layers.extend([deconv, bn, relu, upsample, conv])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


@DETECTORS.register('maskrcnn')
class MaskRCNNModel(FPNFasterRCNN):
    def maskrcnn_pyramid_rcnn_pooling(self, rcnn_feat_maps, rois_batch,
                                      input_size):
        pooled_feats = []
        box_to_levels = []
        # determine which layer to get feat
        # import ipdb
        # ipdb.set_trace()
        roi_level = self.calculate_roi_level(rois_batch)
        for idx, rcnn_feat_map in enumerate(rcnn_feat_maps):
            idx += 2
            mask = roi_level == idx
            rois_batch_per_stage = rois_batch[mask]
            if rois_batch_per_stage.shape[0] == 0:
                continue
            box_to_levels.append(mask.nonzero())
            feat_map_shape = rcnn_feat_map.shape[-2:]
            stride = feat_map_shape[0] / input_size[0]
            pooled_feats.append(
                self.maskrcnn_pooling(rcnn_feat_map, rois_batch_per_stage,
                                      stride))

        # (Important!)Note that you should keep it original order
        pooled_feat = torch.cat(pooled_feats, dim=0)
        box_to_levels = torch.cat(box_to_levels, dim=0).squeeze()
        idx_sorted, order = torch.sort(box_to_levels)
        pooled_feat = pooled_feat[order]
        assert pooled_feat.shape[0] == rois_batch.shape[0]
        return pooled_feat

    def forward(self, feed_dict):
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
                gt_dict[constants.KEY_PRIMARY] = feed_dict[
                    constants.KEY_LABEL_BOXES_2D]
                gt_dict[constants.KEY_CLASSES] = None
                gt_dict[constants.KEY_BOXES_2D] = None
                gt_dict[constants.KEY_CORNERS_2D] = None
                # gt_dict[constants.KEY_CORNERS_VISIBILITY] = None
                # gt_dict[constants.KEY_ORIENTS_V2] = None
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
                auxiliary_dict[constants.KEY_IMAGE_INFO] = feed_dict[
                    constants.KEY_IMAGE_INFO]

                proposals_dict, loss_units, stats = self.target_generators[
                    i].generate_targets(proposals_dict, gt_dict,
                                        auxiliary_dict)

                # note here base_feat (N,C,H,W),rois_batch (N,num_proposals,5)
                proposals = proposals_dict[constants.KEY_PRIMARY]
            rois = box_ops.box2rois(proposals)
            pooled_feat = self.pyramid_rcnn_pooling(rcnn_feat_maps,
                                                    rois.view(-1, 5),
                                                    im_info[0][:2])
            mask_pooled_feat = self.maskrcnn_pyramid_rcnn_pooling(
                rcnn_feat_maps, rois.view(-1, 5), im_info[0][:2])
            keypoint_heatmap = self.keypoint_predictor(mask_pooled_feat)
            keypoint_scores = keypoint_heatmap.view(-1, 56 * 56)
            keypoint_probs = F.softmax(keypoint_scores, dim=-1)

            # shape(N,C,1,1)
            pooled_feat_for_corners = self.feature_extractor.second_stage_feature(
                pooled_feat)
            pooled_feat_for_corners = pooled_feat_for_corners.mean(3).mean(2)

            # keypoint heamap

            rcnn_bbox_preds = self.rcnn_bbox_preds[i](pooled_feat_for_corners)
            rcnn_cls_scores = self.rcnn_cls_preds[i](pooled_feat_for_corners)
            rcnn_corners_preds = self.rcnn_corners_preds[i](
                pooled_feat_for_corners)

            rcnn_dim_preds = self.rcnn_dim_preds[i](pooled_feat_for_corners)

            rcnn_cls_probs = F.softmax(rcnn_cls_scores, dim=1)

            batch_size = rois.shape[0]
            rcnn_cls_scores = rcnn_cls_scores.view(batch_size, -1,
                                                   self.n_classes)
            rcnn_cls_probs = rcnn_cls_probs.view(batch_size, -1,
                                                 self.n_classes)

            rcnn_bbox_preds = rcnn_bbox_preds.view(batch_size, -1, 4)
            rcnn_corners_preds = rcnn_corners_preds.view(
                batch_size, rcnn_bbox_preds.shape[1], -1)

            rcnn_dim_preds = rcnn_dim_preds.view(batch_size, -1, 3)

            # decode for next stage

            keypoint_coder = bbox_coders.build({
                'type':
                constants.KEY_CORNERS_2D_HM
            })
            keypoints = keypoint_coder.decode_keypoint_heatmap(
                proposals, keypoint_probs.view(-1, 4, 56 * 56))

            coder = bbox_coders.build(self.target_generators[i]
                                      .target_generator_config['coder_config'])
            proposals = coder.decode_batch(rcnn_bbox_preds, proposals).detach()
            coder = bbox_coders.build({'type': constants.KEY_DIMS})
            rcnn_dim_preds = coder.decode_batch(
                rcnn_dim_preds, feed_dict[constants.KEY_MEAN_DIMS],
                rcnn_cls_probs).detach()
            coder = bbox_coders.build({
                'type':
                constants.KEY_CORNERS_2D_NEAREST_DEPTH
            })

            # shape(N,C,1,1)

            if self.training:
                loss_units[constants.KEY_CLASSES]['pred'] = rcnn_cls_scores
                loss_units[constants.KEY_BOXES_2D]['pred'] = rcnn_bbox_preds
                loss_units[constants.KEY_DIMS]['pred'] = rcnn_dim_preds
                loss_units[constants.KEY_CORNERS_2D][
                    'pred'] = rcnn_corners_preds
                # loss_units[constants.KEY_CORNERS_VISIBILITY][
                # 'pred'] = rcnn_visibility_preds
                # import ipdb
                # ipdb.set_trace()
                multi_stage_loss_units.append([
                    loss_units[constants.KEY_CLASSES],
                    loss_units[constants.KEY_BOXES_2D],
                    loss_units[constants.KEY_CORNERS_2D],
                    loss_units[constants.KEY_DIMS]
                ])
                multi_stage_stats.append(stats)

        if self.training:
            prediction_dict[constants.KEY_TARGETS] = multi_stage_loss_units
            prediction_dict[constants.KEY_STATS] = multi_stage_stats
        else:
            prediction_dict[constants.KEY_CLASSES] = rcnn_cls_probs

            image_info = feed_dict[constants.KEY_IMAGE_INFO]
            proposals[:, :, ::2] = proposals[:, :, ::
                                             2] / image_info[:, 3].unsqueeze(
                                                 -1).unsqueeze(-1)
            proposals[:, :, 1::2] = proposals[:, :, 1::
                                              2] / image_info[:, 2].unsqueeze(
                                                  -1).unsqueeze(-1)
            rcnn_corners_preds = coder.decode_batch(
                rcnn_corners_preds.detach(), proposals,
                feed_dict[constants.KEY_STEREO_CALIB_P2_ORIG])
            prediction_dict[constants.KEY_CORNERS_2D] = rcnn_corners_preds
            prediction_dict[constants.KEY_BOXES_2D] = proposals
            prediction_dict[constants.KEY_DIMS] = rcnn_dim_preds
            prediction_dict[constants.KEY_CORNERS_2D] = rcnn_corners_preds

            prediction_dict[constants.KEY_KEYPOINTS] = keypoints

        if self.training:
            loss_dict = self.loss(prediction_dict, feed_dict)
            return prediction_dict, loss_dict
        else:
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
        super().init_weights()

        # self.freeze_modules()
        # for param in self.rcnn_depth_preds.parameters():
        # param.requires_grad = True

        # for param in self.third_stage_feature.parameters():
        # param.requires_grad = True

        # self.freeze_bn(self)
        # self.unfreeze_bn(self.third_stage_feature)

    def init_param(self, model_config):
        super().init_param(model_config)
        # all points used for training
        self.use_filter = model_config['use_filter']

    def init_modules(self):
        super().init_modules()
        # combine corners and its visibility
        self.rcnn_corners_preds = nn.ModuleList(
            [nn.Linear(1024, 4 * 8) for _ in range(self.num_stages)])

        # self.third_stage_feature = copy.deepcopy(
            # self.feature_extractor.second_stage_feature)
        # self.rcnn_center_depth_preds = nn.ModuleList(
        # [nn.Linear(1024, 1) for _ in range(self.num_stages)])
        # self.rcnn_visibility_preds = nn.ModuleList(
        # [nn.Linear(1024, 2 * 8) for _ in range(self.num_stages)])

        # not class agnostic for dims
        self.rcnn_dim_preds = nn.ModuleList(
            [nn.Linear(1024, 3) for _ in range(self.num_stages)])

        #  self.rcnn_orient_loss = OrientationLoss()
        # self.rcnn_corners_loss = CornersLoss(
        # use_filter=self.use_filter, training_depth)

        self.maskrcnn_pooling = AdaptiveROIAlign((14, 14), 2)
        self.keypoint_predictor = KeyPointPredictor(256)
        self.rcnn_kp_loss = functools.partial(
            F.cross_entropy, reduce=False, ignore_index=-1)

    def loss(self, prediction_dict, feed_dict):
        """
        assign proposals label and subsample from them
        Then calculate loss
        """
        loss_dict = super().loss(prediction_dict, feed_dict)
        targets = prediction_dict[constants.KEY_TARGETS]
        rcnn_corners_loss = 0
        rcnn_dim_loss = 0

        for stage_ind in range(self.num_stages):
            orient_target = targets[stage_ind][2]
            rcnn_corners_loss = rcnn_corners_loss + common_loss.calc_loss(
                self.rcnn_kp_loss, orient_target, True)

            dim_target = targets[stage_ind][3]
            rcnn_dim_loss = rcnn_dim_loss + common_loss.calc_loss(
                self.rcnn_bbox_loss, dim_target, True)

        loss_dict.update({
            'rcnn_corners_loss': rcnn_corners_loss,
            #  'rcnn_dim_loss': rcnn_dim_loss
        })

        return loss_dict
