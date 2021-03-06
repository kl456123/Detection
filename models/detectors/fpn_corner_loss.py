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
from models.losses.corners_3d_loss import Corners3DLoss
from models.losses.corners_loss import CornersLoss

from utils.registry import DETECTORS
from utils import box_ops

from target_generators.target_generator import TargetGenerator
from models import feature_extractors
from models import detectors

from utils import batch_ops
import bbox_coders
from core.utils.analyzer import Analyzer

from utils import geometry_utils


@DETECTORS.register('fpn_corner_loss')
class FPNGRNetModel(FPNFasterRCNN):
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
                gt_dict[constants.KEY_CORNERS_3D_GRNET] = None
                # gt_dict[constants.KEY_CORNERS_VISIBILITY] = None
                # gt_dict[constants.KEY_ORIENTS_V2] = None
                # gt_dict[constants.KEY_DIMS] = None

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
                # auxiliary_dict[constants.KEY_MEAN_DIMS] = feed_dict[
                # constants.KEY_MEAN_DIMS]
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
            pooled_feat = self.feature_extractor.second_stage_feature(
                pooled_feat)
            pooled_feat = pooled_feat.mean(3).mean(2)

            rcnn_bbox_preds = self.rcnn_bbox_preds[i](pooled_feat)
            rcnn_cls_scores = self.rcnn_cls_preds[i](pooled_feat)
            rcnn_corners_preds = self.rcnn_corners_preds[i](pooled_feat)
            # rcnn_visibility_preds = self.rcnn_visibility_preds[i](pooled_feat)
            # rcnn_dim_preds = self.rcnn_dim_preds[i](pooled_feat)

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
            # if not self.class_agnostic_3d:
            # if self.training:
            # rcnn_dim_preds = self.squeeze_bbox_preds(
            # rcnn_dim_preds,
            # loss_units[constants.KEY_CLASSES]['target'].view(-1),
            # out_c=3)
            # else:
            # rcnn_dim_preds = self.squeeze_bbox_preds(
            # rcnn_dim_preds,
            # rcnn_cls_probs.argmax(dim=-1).view(-1),
            # out_c=3)

            rcnn_bbox_preds = rcnn_bbox_preds.view(batch_size, -1, 4)
            rcnn_corners_preds = rcnn_corners_preds.view(
                batch_size, rcnn_bbox_preds.shape[1], -1)
            # rcnn_visibility_preds = rcnn_visibility_preds.view(
            # batch_size, rcnn_bbox_preds.shape[1], -1)
            # rcnn_dim_preds = rcnn_dim_preds.view(batch_size, -1, 3)

            if self.training:
                loss_units[constants.KEY_CLASSES]['pred'] = rcnn_cls_scores
                loss_units[constants.KEY_BOXES_2D]['pred'] = rcnn_bbox_preds
                # loss_units[constants.KEY_DIMS]['pred'] = rcnn_dim_preds
                loss_units[constants.KEY_CORNERS_3D_GRNET][
                    'pred'] = rcnn_corners_preds
                # loss_units[constants.KEY_CORNERS_VISIBILITY][
                # 'pred'] = rcnn_visibility_preds
                # import ipdb
                # ipdb.set_trace()
                multi_stage_loss_units.append([
                    loss_units[constants.KEY_CLASSES],
                    loss_units[constants.KEY_BOXES_2D],
                    loss_units[constants.KEY_CORNERS_3D_GRNET],
                    # loss_units[constants.KEY_DIMS]
                ])
                multi_stage_stats.append(stats)
                # coder = bbox_coders.build({'type': constants.KEY_DIMS})
                # rcnn_dim_preds = coder.decode_batch(
                # rcnn_dim_preds, feed_dict[constants.KEY_MEAN_DIMS],
                # rcnn_cls_probs)
            else:
                # coder = bbox_coders.build({'type': constants.KEY_DIMS})
                # rcnn_dim_preds = coder.decode_batch(
                # rcnn_dim_preds, feed_dict[constants.KEY_MEAN_DIMS],
                # rcnn_cls_probs).detach()
                # decode for next stage
                coder = bbox_coders.build({
                    'type':
                    constants.KEY_CORNERS_3D_BETTER
                })
                rcnn_corners_preds = coder.decode_batch(
                    rcnn_corners_preds.detach(), proposals,
                    feed_dict[constants.KEY_STEREO_CALIB_P2])
                coder = bbox_coders.build(
                    self.target_generators[i]
                    .target_generator_config['coder_config'])
                proposals = coder.decode_batch(rcnn_bbox_preds,
                                               proposals).detach()

        if self.training:
            prediction_dict[constants.KEY_TARGETS] = multi_stage_loss_units
            prediction_dict[constants.KEY_STATS] = multi_stage_stats
            prediction_dict[constants.KEY_PROPOSALS] = proposals
            # prediction_dict[constants.KEY_DIMS] = rcnn_dim_preds
        else:
            prediction_dict[constants.KEY_CLASSES] = rcnn_cls_probs

            image_info = feed_dict[constants.KEY_IMAGE_INFO]
            proposals[:, :, ::2] = proposals[:, :, ::
                                             2] / image_info[:, 3].unsqueeze(
                                                 -1).unsqueeze(-1)
            proposals[:, :, 1::2] = proposals[:, :, 1::
                                              2] / image_info[:, 2].unsqueeze(
                                                  -1).unsqueeze(-1)
            # rcnn_corners_preds = coder.decode_batch(
            # rcnn_corners_preds.detach(), proposals)

            # import ipdb
            # ipdb.set_trace()
            # rcnn_corners_preds = torch.bmm(
            # feed_dict[constants.KEY_STEREO_CALIB_P2_ORIG],
            # rcnn_corners_preds)
            # assert rcnn_corners_preds.shape[0] == 1
            # rcnn_corners_preds = geometry_utils.torch_points_3d_to_points_2d(
            # rcnn_corners_preds[0].view(-1, 3),
            # feed_dict[constants.KEY_STEREO_CALIB_P2_ORIG][0]).view(-1, 8,
            # 2)
            prediction_dict[constants.KEY_CORNERS_2D] = rcnn_corners_preds
            prediction_dict[constants.KEY_BOXES_2D] = proposals
            # prediction_dict[constants.KEY_DIMS] = rcnn_dim_preds

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

        # if self.freeze_2d:
        # self.freeze_modules()
        # for param in self.rcnn_corners_preds.parameters():
        # param.requires_grad = True

        # self.freeze_bn(self)

    def init_param(self, model_config):
        super().init_param(model_config)
        self.class_agnostic_3d = False
        # self.freeze_2d = model_config.get('freeze_2d', False)

    def init_modules(self):
        super().init_modules()
        # combine corners and its visibility
        self.rcnn_corners_preds = nn.ModuleList(
            [nn.Linear(1024, 1 + 2 + 1 + 3) for _ in range(self.num_stages)])
        # self.rcnn_visibility_preds = nn.ModuleList(
        # [nn.Linear(1024, 2 * 8) for _ in range(self.num_stages)])

        # not class agnostic for dims
        # if not self.class_agnostic_3d:
        # self.rcnn_dim_preds = nn.ModuleList([
        # nn.Linear(1024, self.n_classes * 3)
        # for _ in range(self.num_stages)
        # ])
        # else:
        # self.rcnn_dim_preds = nn.ModuleList(
        # [nn.Linear(1024, 3) for _ in range(self.num_stages)])

        #  self.rcnn_orient_loss = OrientationLoss()
        self.l1_loss = nn.L1Loss(reduction='none')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')

    def loss(self, prediction_dict, feed_dict):
        """
        assign proposals label and subsample from them
        Then calculate loss
        """
        loss_dict = super().loss(prediction_dict, feed_dict)
        targets = prediction_dict[constants.KEY_TARGETS]

        proposals = prediction_dict[constants.KEY_PROPOSALS]
        p2 = feed_dict[constants.KEY_STEREO_CALIB_P2]
        image_info = feed_dict[constants.KEY_IMAGE_INFO]
        corners_2d_loss = 0
        center_depth_loss = 0
        location_loss = 0
        mean_dims = torch.tensor([1.8, 1.8, 3.7]).type_as(proposals)

        for stage_ind in range(self.num_stages):

            # dims loss
            # dim_target = targets[stage_ind][3]
            # dim_loss = common_loss.calc_loss(self.rcnn_bbox_loss, dim_target,
            # True)

            corners_target = targets[stage_ind][2]
            # dims_preds = targets[stage_ind][3]['pred']

            preds = corners_target['pred']
            N, M = preds.shape[:2]
            targets = corners_target['target']
            weights = corners_target['weight']

            # gt
            corners_2d_gt = targets[:, :, :16]
            location_gt = targets[:, :, 16:19]
            dims_gt = targets[:, :, 19:]
            center_depth_gt = location_gt[:, :, -1:]

            center_depth_preds = preds[:, :, :1]
            center_2d_deltas_preds = preds[:, :, 1:3]
            ry_preds = preds[:, :, 3:4]
            # import ipdb
            # ipdb.set_trace()
            dims_preds = torch.exp(preds[:, :, 4:]) * mean_dims
            # convert to corners 2d

            # convert to location
            # decode center_2d
            proposals_xywh = geometry_utils.torch_xyxy_to_xywh(proposals)
            center_2d_preds = (
                center_2d_deltas_preds * proposals_xywh[:, :, 2:] +
                proposals_xywh[:, :, :2])

            location_preds = []
            for batch_ind in range(N):
                location_preds.append(
                    geometry_utils.torch_points_2d_to_points_3d(
                        center_2d_preds[batch_ind],
                        center_depth_preds[batch_ind], p2[batch_ind]))
            location_preds = torch.stack(location_preds, dim=0)

            # concat
            # import ipdb
            # ipdb.set_trace()
            boxes_3d_preds = torch.cat(
                [location_preds, dims_preds.detach(), ry_preds], dim=-1)
            corners_2d_preds = []
            for batch_ind in range(N):
                corners_2d_preds.append(
                    geometry_utils.torch_boxes_3d_to_corners_2d(
                        boxes_3d_preds[batch_ind], p2[batch_ind]))
            corners_2d_preds = torch.stack(
                corners_2d_preds, dim=0).view(N, M, -1)

            weights = weights.unsqueeze(-1)

            # import ipdb
            # ipdb.set_trace()
            # corners depth loss and center depth loss

            center_depth_loss = self.l1_loss(center_depth_preds,
                                             center_depth_gt) * weights

            # location loss
            location_loss = self.l1_loss(location_preds, location_gt) * weights

            # import ipdb
            # ipdb.set_trace()
            # dims loss
            dims_loss = self.smooth_l1_loss(dims_preds, dims_gt) * weights

            # proj 2d loss
            zeros = torch.zeros_like(image_info[:, 0])
            image_shape = torch.stack(
                [zeros, zeros, image_info[:, 1], image_info[:, 0]], dim=-1)
            image_shape = image_shape.type_as(corners_2d_gt).view(-1, 4)
            image_filter = geometry_utils.torch_window_filter(
                corners_2d_gt.contiguous().view(N, -1, 2),
                image_shape,
                deltas=200).float().view(N, M, -1)

            corners_2d_loss = self.l1_loss(corners_2d_preds,
                                           corners_2d_gt) * weights
            corners_2d_loss = (corners_2d_loss.view(N, M, 8, 2) *
                               image_filter.unsqueeze(-1)).view(N, M, -1)

        loss_dict.update({
            # 'global_corners_loss': global_corners_loss * 10,
            'corners_2d_loss': corners_2d_loss,
            'center_depth_loss': center_depth_loss * 10,
            'location_loss': location_loss * 10,
            # 'rcnn_corners_loss': rcnn_corners_loss,
            'dims_loss': dims_loss
        })

        return loss_dict
