# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.nn.functional as F

from lib.model.roi_layers import ROIAlign

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
    def __init__(self, inplane, output=8):
        super().__init__()

        layers = []
        for i in range(6):
            layers.append(conv3x3_bn_relu(inplane, 256))
            inplane = 256

        # upsample
        deconv = nn.ConvTranspose2d(256, 256, 2, 2, 0)
        bn = nn.BatchNorm2d(256)
        relu = nn.ReLU()
        # upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        conv = nn.Conv2d(256, output, 1, 1, 0)
        layers.extend([deconv, bn, relu, conv])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


@DETECTORS.register('mobileye')
class FPNCornersModel(FPNFasterRCNN):
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
                gt_dict[constants.KEY_MOBILEYE] = None
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

            # shape(N,C,1,1)
            pooled_feat_for_corners = self.feature_extractor.second_stage_feature(
                pooled_feat)
            # pooled_feat_for_keypoint = F.upsample_bilinear(
                # pooled_feat_for_corners, size=(14, 14))
            keypoint_map = self.keypoint_predictor(pooled_feat)
            # keypoint_map = self.rcnn_keypoint_preds(pooled_feat_for_keypoint)
            keypoint_map = keypoint_map.mean(-2)
            # keypoint_map = F.softmax(keypoint_map, dim=-1)

            pooled_feat_for_corners = pooled_feat_for_corners.mean(3).mean(2)

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

            # not class_agnostic for dims
            # import ipdb
            # ipdb.set_trace()
            if not self.class_agnostic_3d:
                if self.training:
                    rcnn_dim_preds = self.squeeze_bbox_preds(
                        rcnn_dim_preds,
                        loss_units[constants.KEY_CLASSES]['target'].view(-1),
                        out_c=3)
                else:
                    rcnn_dim_preds = self.squeeze_bbox_preds(
                        rcnn_dim_preds,
                        rcnn_cls_probs.argmax(dim=-1).view(-1),
                        out_c=3)

            rcnn_bbox_preds = rcnn_bbox_preds.view(batch_size, -1, 4)
            rcnn_corners_preds = rcnn_corners_preds.view(
                batch_size, rcnn_bbox_preds.shape[1], -1)

            # rcnn_depth_preds = rcnn_depth_preds.view(
            # batch_size, rcnn_bbox_preds.shape[1], -1)
            # rcnn_center_depth_preds = rcnn_center_depth_preds.view(
            # batch_size, rcnn_bbox_preds.shape[1], -1)
            # concat them(depth and corners)
            # rcnn_corners_preds = torch.cat(
            # [rcnn_corners_preds, rcnn_depth_preds], dim=-1)

            # # append center depth
            # rcnn_corners_preds = torch.cat(
            # [rcnn_corners_preds, rcnn_center_depth_preds], dim=-1)

            # rcnn_visibility_preds = rcnn_visibility_preds.view(
            # batch_size, rcnn_bbox_preds.shape[1], -1)
            rcnn_dim_preds = rcnn_dim_preds.view(batch_size, -1, 3)

            # decode for next stage

            coder = bbox_coders.build({'type': constants.KEY_DIMS})
            rcnn_dim_preds = coder.decode_batch(
                rcnn_dim_preds, feed_dict[constants.KEY_MEAN_DIMS],
                rcnn_cls_probs).detach()

            # rcnn_corners_preds = coder.decode_batch(
            # rcnn_corners_preds.detach(), proposals)

            # import ipdb
            # ipdb.set_trace()
            # if self.training_depth:
            # # predict for depth
            # rois = box_ops.box2rois(proposals)
            # pooled_feat_for_depth = self.pyramid_rcnn_pooling(
            # rcnn_feat_maps, rois.view(-1, 5), im_info[0][:2])

            # shape(N,C,1,1)
            # pooled_feat_for_depth = self.third_stage_feature(pooled_feat)
            # pooled_feat_for_depth = pooled_feat_for_depth.mean(3).mean(2)
            # rcnn_depth_preds = self.rcnn_depth_preds[i](pooled_feat_for_depth)

            # encode
            # rcnn_depth_preds = 1 / (rcnn_depth_preds.sigmoid() + 1e-6) - 1
            # rcnn_depth_preds = rcnn_depth_preds.view(
            # batch_size, rcnn_bbox_preds.shape[1], -1)

            # # concat them(depth and corners)
            # rcnn_corners_preds = self.fuse_corners_and_depth(
            # rcnn_corners_preds, rcnn_depth_preds)
            # rcnn_corners_preds = torch.cat(
            # [rcnn_corners_preds, rcnn_depth_preds], dim=-1)

            # # # append center depth
            # rcnn_corners_preds = torch.cat(
            # [rcnn_corners_preds, rcnn_center_depth_preds], dim=-1)

            if self.training:
                loss_units[constants.KEY_CLASSES]['pred'] = rcnn_cls_scores
                loss_units[constants.KEY_BOXES_2D]['pred'] = rcnn_bbox_preds
                loss_units[constants.KEY_DIMS]['pred'] = rcnn_dim_preds
                loss_units[constants.KEY_MOBILEYE]['pred'] = rcnn_corners_preds
                multi_stage_loss_units.append([
                    loss_units[constants.KEY_CLASSES],
                    loss_units[constants.KEY_BOXES_2D],
                    loss_units[constants.KEY_MOBILEYE],
                    loss_units[constants.KEY_DIMS]
                ])
                multi_stage_stats.append(stats)
        coder = bbox_coders.build({'type': constants.KEY_MOBILEYE})
        rcnn_corners_preds = coder.decode_batch(
            rcnn_corners_preds.detach(), proposals, keypoint_map.detach())
        prediction_dict[constants.KEY_CORNERS_2D] = rcnn_corners_preds
        prediction_dict[constants.KEY_KEYPOINTS_HEATMAP] = keypoint_map
        # if self.training:
        # corners_2d_gt = coder.decode_batch(
        # loss_units[constants.KEY_MOBILEYE]['target'], proposals)
        # prediction_dict['corners_2d_gt'] = corners_2d_gt
        if self.training:
            prediction_dict[constants.KEY_TARGETS] = multi_stage_loss_units
            prediction_dict[constants.KEY_STATS] = multi_stage_stats
        else:
            prediction_dict[constants.KEY_CLASSES] = rcnn_cls_probs
            coder = bbox_coders.build(self.target_generators[i]
                                      .target_generator_config['coder_config'])
            proposals = coder.decode_batch(rcnn_bbox_preds, proposals).detach()

            image_info = feed_dict[constants.KEY_IMAGE_INFO]
            proposals[:, :, ::2] = proposals[:, :, ::
                                             2] / image_info[:, 3].unsqueeze(
                                                 -1).unsqueeze(-1)
            proposals[:, :, 1::2] = proposals[:, :, 1::
                                              2] / image_info[:, 2].unsqueeze(
                                                  -1).unsqueeze(-1)

            rcnn_corners_preds[:, :, :,
                               0] = rcnn_corners_preds[:, :, :,
                                                       0] / image_info[:, None,
                                                                       None, 3]
            rcnn_corners_preds[:, :, :,
                               1] = rcnn_corners_preds[:, :, :,
                                                       1] / image_info[:, None,
                                                                       None, 2]
            prediction_dict[constants.KEY_BOXES_2D] = proposals
            # prediction_dict[constants.KEY_DIMS] = rcnn_dim_preds
            # prediction_dict[constants.KEY_CORNERS_2D] = rcnn_corners_preds

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
        self.class_agnostic_3d = False
        # all points used for training
        # self.use_filter = model_config['use_filter']
        # self.training_depth = model_config['training_depth']

    def init_modules(self):
        super().init_modules()
        # combine corners and its visibility
        in_channels = 1024
        self.rcnn_corners_preds = nn.ModuleList(
            [nn.Linear(in_channels, 11) for _ in range(self.num_stages)])

        # self.rcnn_keypoint_preds = nn.Conv2d(in_channels, 1, 1, 1, 0)
        self.keypoint_predictor = KeyPointPredictor(256, 1)
        # self.rcnn_depth_preds = nn.ModuleList(
        # [nn.Linear(in_channels, 1 * 8 + 1) for _ in range(self.num_stages)])

        # self.third_stage_feature = copy.deepcopy(
        # self.feature_extractor.second_stage_feature)
        # self.rcnn_center_depth_preds = nn.ModuleList(
        # [nn.Linear(1024, 1) for _ in range(self.num_stages)])
        # self.rcnn_visibility_preds = nn.ModuleList(
        # [nn.Linear(1024, 2 * 8) for _ in range(self.num_stages)])

        # not class agnostic for dims
        if not self.class_agnostic_3d:
            self.rcnn_dim_preds = nn.ModuleList([
                nn.Linear(in_channels, self.n_classes * 3)
                for _ in range(self.num_stages)
            ])
        else:
            self.rcnn_dim_preds = nn.ModuleList(
                [nn.Linear(in_channels, 3) for _ in range(self.num_stages)])

        #  self.rcnn_orient_loss = OrientationLoss()
        # self.rcnn_corners_loss = CornersLoss(
        # use_filter=self.use_filter, training_depth=self.training_depth)
        self.l1_loss = nn.L1Loss(reduction='none')

    def get_slope(self, bottom_corners):
        """
        Args:
            bottom_corners: shape(N, M, 4, 2)
        Returns:
            pass
        """
        # import ipdb
        # ipdb.set_trace()
        start_corners = bottom_corners
        end_corners = bottom_corners[:, :, [1, 2, 3, 0]]
        direction = end_corners - start_corners
        norm = torch.norm(direction, dim=-1).unsqueeze(-1)
        return direction / (norm + 1e-7)

    def loss(self, prediction_dict, feed_dict):
        """
        assign proposals label and subsample from them
        Then calculate loss
        """
        loss_dict = super().loss(prediction_dict, feed_dict)
        targets = prediction_dict[constants.KEY_TARGETS]
        keypoint_map = prediction_dict[constants.KEY_KEYPOINTS_HEATMAP]
        # corners_2d = prediction_dict[constants.KEY_CORNERS_2D]
        # corners_2d_gt = prediction_dict['corners_2d_gt']
        # rcnn_corners_loss = 0
        # rcnn_dim_loss = 0
        # slope_loss = 0
        # import ipdb
        # ipdb.set_trace()

        for stage_ind in range(self.num_stages):
            mobileye_target = targets[stage_ind][2]

            target = mobileye_target['target']
            pred = mobileye_target['pred']
            weight = mobileye_target['weight']
            num_pos = weight[weight > 0].float().sum()
            num_pos = num_pos.clamp(min=1)
            # import ipdb
            # ipdb.set_trace()
            height_pred = pred[:, :, 6:9]
            height_gt = target[:, :, 6:9]
            corners_pred = pred[:, :, :6]
            corners_gt = target[:, :, :6]
            ratio = target[:, :, 9]
            resolution = keypoint_map.shape[-1]
            ratio = ratio * resolution
            ratio = ratio.clamp(max=resolution - 1)
            keypoint_loss = self.rcnn_cls_loss(
                keypoint_map.view(-1, resolution),
                ratio.long().view(-1)) * weight.view(-1)
            # visibility_gt = target[:, :, 9]
            # visibility_pred = pred[:, :, 9:11]

            N, M = corners_pred.shape[:2]
            corners_loss = self.l1_loss(
                corners_pred, corners_gt) * weight.unsqueeze(-1).view(
                    N, M, -1)
            corners_loss = corners_loss
            height_loss = self.l1_loss(height_pred,
                                       height_gt) * weight.unsqueeze(-1)
            mobileye_loss = torch.cat([corners_loss, height_loss], dim=-1)

            # mobileye_loss = self.l1_loss(pred, target) * weight.unsqueeze(-1)

            # get slope for pred and gt

            # bottom_corners_pred = corners_2d[:, :, :4]
            # bottom_corners_gt = corners_2d_gt[:, :, :4]
            # slope_pred = self.get_slope(bottom_corners_pred)
            # slope_gt = self.get_slope(bottom_corners_gt)

            # slope_loss = slope_loss + self.l1_loss(
            # slope_pred, slope_gt) * weight.unsqueeze(-1).unsqueeze(-1)
            # import ipdb
            # ipdb.set_trace()
            # visibility_loss = self.rcnn_cls_loss(
            # visibility_pred.view(-1, 2),
            # visibility_gt.long().view(-1)) * weight.view(-1)
            # rcnn_corners_loss = rcnn_corners_loss + common_loss.calc_loss(
            # self.rcnn_corners_loss, orient_target, True)

            # dim_target = targets[stage_ind][3]
            # rcnn_dim_loss = rcnn_dim_loss + common_loss.calc_loss(
            # self.rcnn_bbox_loss, dim_target, True)

        loss_dict.update({
            'mobileye_loss': mobileye_loss.sum() / num_pos,
            'keypoint_loss': keypoint_loss.sum() / num_pos
            # 'visibility_loss': visibility_loss.sum() / num_pos
            # 'slope_loss': slope_loss.sum() / num_pos
            # 'rcnn_corners_loss': rcnn_corners_loss,
            #  'rcnn_dim_loss': rcnn_dim_loss
        })

        return loss_dict
