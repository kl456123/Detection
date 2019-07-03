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
from models.losses.keypoint_loss import KeyPointLoss

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


@DETECTORS.register('maskrcnn')
class MaskRCNNModel(FPNFasterRCNN):
    def init_param(self, model_config):
        super().init_param(model_config)
        self.mask_enable = False
        self.depth_enable = True

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

    def check_proposals(self, proposals):
        # import ipdb
        # ipdb.set_trace()
        ws = proposals[:, :, 2] - proposals[:, :, 0]
        hs = proposals[:, :, 3] - proposals[:, :, 1]
        assert (ws[ws > 0].min() >= 2) & (hs[hs > 0].min() >= 2)

    def forward(self, feed_dict):
        # import ipdb
        # ipdb.set_trace()

        # self.crop_instance_map_gt(feed_dict)

        im_info = feed_dict[constants.KEY_IMAGE_INFO]
        batch_size = im_info.shape[0]

        prediction_dict = {}

        # base model
        rpn_feat_maps, rcnn_feat_maps = self.feature_extractor.first_stage_feature(
            feed_dict[constants.KEY_IMAGE])
        feed_dict.update({'rpn_feat_maps': rpn_feat_maps})

        # rpn model
        prediction_dict.update(self.rpn_model.forward(feed_dict))
        proposals = prediction_dict['proposals']
        self.check_proposals(proposals)
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
                if self.depth_enable:
                    gt_dict[constants.KEY_CORNERS_3D_GRNET] = None
                # gt_dict[constants.KEY_INSTANCES_MASK] = None
                # gt_dict[constants.KEY_KEYPOINTS] = None
                # gt_dict[constants.KEY_CORNERS_VISIBILITY] = None
                # gt_dict[constants.KEY_ORIENTS_V2] = None
                # gt_dict[constants.KEY_DIMS] = None

                # auxiliary_dict(used for encoding)
                auxiliary_dict = {}
                if self.depth_enable:
                    auxiliary_dict[constants.KEY_STEREO_CALIB_P2] = feed_dict[
                        constants.KEY_STEREO_CALIB_P2]
                    auxiliary_dict[constants.KEY_BOXES_3D] = feed_dict[
                        constants.KEY_LABEL_BOXES_3D]
                auxiliary_dict[constants.KEY_BOXES_2D] = feed_dict[
                    constants.KEY_LABEL_BOXES_2D]
                auxiliary_dict[constants.KEY_CLASSES] = feed_dict[
                    constants.KEY_LABEL_CLASSES]

                auxiliary_dict[constants.KEY_NUM_INSTANCES] = feed_dict[
                    constants.KEY_NUM_INSTANCES]
                auxiliary_dict[constants.KEY_PROPOSALS] = proposals
                # auxiliary_dict[constants.KEY_MEAN_DIMS] = feed_dict[
                # constants.KEY_MEAN_DIMS]
                auxiliary_dict[constants.KEY_IMAGE_INFO] = feed_dict[
                    constants.KEY_IMAGE_INFO]
                # auxiliary_dict[constants.KEY_INSTANCES_MASK] = feed_dict[
                # constants.KEY_INSTANCES_MASK]

                # auxiliary_dict[constants.KEY_KEYPOINTS] = feed_dict[
                # constants.KEY_KEYPOINTS]

                proposals_dict, loss_units, stats = self.target_generators[
                    i].generate_targets(proposals_dict, gt_dict,
                                        auxiliary_dict)

                # note here base_feat (N,C,H,W),rois_batch (N,num_proposals,5)
                proposals = proposals_dict[constants.KEY_PRIMARY]

            # check proposals
            self.check_proposals(proposals)
            rois = box_ops.box2rois(proposals)
            pooled_feat = self.pyramid_rcnn_pooling(rcnn_feat_maps,
                                                    rois.view(-1, 5),
                                                    im_info[0][:2])

            pooled_feat_for_corners = self.feature_extractor.second_stage_feature(
                pooled_feat)
            pooled_feat_for_corners = pooled_feat_for_corners.mean(3).mean(2)

            # keypoint heamap

            rcnn_bbox_preds = self.rcnn_bbox_preds[i](pooled_feat_for_corners)
            rcnn_cls_scores = self.rcnn_cls_preds[i](pooled_feat_for_corners)

            rcnn_dim_preds = self.rcnn_dim_preds[i](pooled_feat_for_corners)
            rcnn_depth_preds = self.rcnn_depth_preds[i](
                pooled_feat_for_corners)

            rcnn_cls_probs = F.softmax(rcnn_cls_scores, dim=1)

            rcnn_cls_scores = rcnn_cls_scores.view(batch_size, -1,
                                                   self.n_classes)
            rcnn_cls_probs = rcnn_cls_probs.view(batch_size, -1,
                                                 self.n_classes)

            rcnn_bbox_preds = rcnn_bbox_preds.view(batch_size, -1, 4)

            rcnn_dim_preds = rcnn_dim_preds.view(batch_size, -1, 3)

            rcnn_depth_preds = rcnn_depth_preds.view(batch_size, -1, 1)
            if not self.training:
                coder = bbox_coders.build(
                    self.target_generators[i]
                    .target_generator_config['coder_config'])
                proposals = coder.decode_batch(rcnn_bbox_preds,
                                               proposals).detach()
                rois = box_ops.box2rois(proposals)
                mask_pooled_feat = self.maskrcnn_pyramid_rcnn_pooling(
                    rcnn_feat_maps, rois.view(-1, 5), im_info[0][:2])
            else:
                mask_pooled_feat = self.maskrcnn_pyramid_rcnn_pooling(
                    rcnn_feat_maps, rois.view(-1, 5), im_info[0][:2])
            resolution = 28
            batch_size = rois.shape[0]
            if self.depth_enable:
                depth_map = self.depth_map_predictor(mask_pooled_feat)
                depth_map = depth_map.view(batch_size, -1,
                                           resolution * resolution)

            if self.mask_enable:
                instance_map = self.instance_map_predictor(mask_pooled_feat)
                instance_map = instance_map.view(batch_size, -1,
                                                 resolution * resolution * 2)

            # keypoint_scores = keypoint_heatmap.view(-1, 56 * 56)
            # keypoint_probs = F.softmax(keypoint_scores, dim=-1)

            # shape(N,C,1,1)

            if self.training:
                loss_units[constants.KEY_CLASSES]['pred'] = rcnn_cls_scores
                loss_units[constants.KEY_BOXES_2D]['pred'] = rcnn_bbox_preds
                # loss_units[constants.KEY_DIMS]['pred'] = rcnn_dim_preds
                # loss_units[constants.KEY_KEYPOINTS]['pred'] = keypoint_heatmap
                weights = loss_units[constants.KEY_BOXES_2D]['weight']
                if self.depth_enable:
                    loss_units[constants.KEY_DEPTHMAP] = {}
                    loss_units[constants.KEY_DEPTHMAP]['pred'] = depth_map

                    # depth map loss units
                    loss_units[constants.KEY_DEPTHMAP]['weight'] = weights
                    loss_units[constants.KEY_CORNERS_3D_GRNET][
                        'pred'] = rcnn_depth_preds
                    # import ipdb
                    # ipdb.set_trace()
                    fg_map = self.crop_instance_map(
                        proposals,
                        feed_dict[constants.KEY_LABEL_INSTANCES_MASK],
                        weights,
                        auxiliary_dict[constants.KEY_MATCH],
                        mask_size=resolution)
                    loss_units[constants.KEY_DEPTHMAP]['fg_weight'] = fg_map

                    proposals_depth_map = self.crop_depth_map(
                        proposals,
                        feed_dict[constants.KEY_LABEL_DEPTHMAP],
                        weights,
                        mask_size=resolution)
                    loss_units[constants.KEY_DEPTHMAP][
                        'target'] = proposals_depth_map

                if self.mask_enable:
                    # instance map loss units
                    proposals_instance_map = self.crop_instance_map(
                        proposals,
                        feed_dict[constants.KEY_LABEL_INSTANCES_MASK],
                        weights,
                        auxiliary_dict[constants.KEY_MATCH],
                        feed_dict[constants.KEY_IMAGE_INFO],
                        mask_size=resolution)
                    instance_mask_units = {
                        'pred': instance_map,
                        'weight': weights,
                        'target': proposals_instance_map
                    }
                    loss_units[constants.
                               KEY_LABEL_INSTANCES_MASK] = instance_mask_units

                multi_stage_loss_units.append([
                    loss_units[constants.KEY_CLASSES],
                    loss_units[constants.KEY_BOXES_2D],
                ])

                if self.depth_enable:
                    multi_stage_loss_units[0].extend([
                        loss_units[constants.KEY_DEPTHMAP],
                        loss_units[constants.KEY_CORNERS_3D_GRNET]
                    ])
                if self.mask_enable:
                    multi_stage_loss_units[0].append(
                        loss_units[constants.KEY_LABEL_INSTANCES_MASK])
                multi_stage_stats.append(stats)

        if self.training:
            prediction_dict[constants.KEY_TARGETS] = multi_stage_loss_units
            prediction_dict[constants.KEY_STATS] = multi_stage_stats
        else:
            prediction_dict[constants.KEY_CLASSES] = rcnn_cls_probs

            image_info = feed_dict[constants.KEY_IMAGE_INFO]
            image_info = image_info.unsqueeze(1).unsqueeze(1)
            proposals[:, :, ::2] = proposals[:, :, ::2] / image_info[..., 3]
            proposals[:, :, 1::2] = proposals[:, :, 1::2] / image_info[..., 2]
            prediction_dict[constants.KEY_BOXES_2D] = proposals
            prediction_dict[
                constants.KEY_DEPTHMAP] = depth_map + rcnn_depth_preds

        if self.training:
            loss_dict = self.loss(prediction_dict, feed_dict)
            return prediction_dict, loss_dict
        else:
            return prediction_dict

    def crop_instance_map_gt(self, feed_dict):
        # import ipdb
        # ipdb.set_trace()
        instance_map = feed_dict[constants.KEY_LABEL_INSTANCES_MASK]
        label_boxes_2d = feed_dict[constants.KEY_LABEL_BOXES_2D]
        num_instances = feed_dict[constants.KEY_NUM_INSTANCES]
        image_info = feed_dict[constants.KEY_IMAGE_INFO]
        x = label_boxes_2d[..., ::2] / image_info[:, 3][..., None, None]
        y = label_boxes_2d[..., 1::2] / image_info[:, 2][..., None, None]
        orig_boxes_2d = torch.stack(
            [x[..., 0], y[..., 0], x[..., 1], y[..., 1]], dim=-1).long()

        for batch_ind in range(label_boxes_2d.shape[0]):
            for box_id in range(num_instances):
                box = orig_boxes_2d[batch_ind][box_id]

                cropped_mask = instance_map[batch_ind:batch_ind + 1, :, box[1]:
                                            box[3], box[0]:box[2]]

    def crop_instance_map(self,
                          proposals,
                          depth_map,
                          weights,
                          match,
                          image_info=None,
                          mask_size=28,
                          device='cuda'):
        """
        Generate depth map for each proposal
        Args:
            depth_map: shape(N, 1, H, W)
            proposals: shape(N, M, 4)
        Returns:
            proposals_depth_map: shape(N, M, mask_size*mask_size)
        """
        # import ipdb
        # ipdb.set_trace()
        # import matplotlib.pyplot as plt
        # plt.imshow(depth_map.cpu().numpy()[0, 0])

        N, M = proposals.shape[:2]
        H, W = depth_map.shape[-2:]
        if image_info is not None:
            x = proposals[..., ::2] / image_info[:, 3][..., None, None]
            y = proposals[..., 1::2] / image_info[:, 2][..., None, None]
            proposals = torch.stack(
                [x[..., 0], y[..., 0], x[..., 1], y[..., 1]], dim=-1)
        proposals = proposals.long()
        depth_targets = []
        # zero refers to bg
        match = match + 1
        for batch_ind in range(N):
            # proposals
            proposals_single_image = proposals[batch_ind]
            pos_proposals_single_image = proposals_single_image[
                weights[batch_ind] > 0]
            # match
            match_single_image = match[batch_ind]
            pos_match_single_image = match_single_image[weights[batch_ind] > 0]

            for proposals_ind in range(pos_proposals_single_image.shape[0]):
                box = pos_proposals_single_image[proposals_ind]
                cropped_mask = depth_map[batch_ind:batch_ind + 1, :, box[1]:
                                         box[3], box[0]:box[2]]
                # plt.imshow(cropped_mask.cpu().numpy()[0, 0])
                # plt.show()
                instance_mask = torch.zeros_like(cropped_mask)
                instance_mask[cropped_mask == pos_match_single_image[
                    proposals_ind]] = 1

                feat_shape = instance_mask.shape[-2:]
                if feat_shape[0] == 0 or feat_shape[1] == 0:
                    import ipdb
                    ipdb.set_trace()
                instance_mask = F.upsample_bilinear(
                    instance_mask.float(), size=mask_size)
                instance_mask[instance_mask > 0] = 1
                # instance_mask[instance_mask < 0.5] = 0
                # plt.imshow(instance_mask.cpu().numpy()[0, 0])
                # plt.show()

                depth_targets.append(instance_mask.long())

        # import ipdb
        # ipdb.set_trace()
        if len(depth_targets) == 0:
            depth_targets = torch.zeros(1, 0, 28, 28).type_as(depth_map)
        else:
            depth_targets = torch.cat(depth_targets, dim=1)
        return depth_targets.view(-1, mask_size * mask_size)

    def crop_depth_map(self,
                       proposals,
                       depth_map,
                       weights,
                       mask_size=28,
                       device='cuda'):
        """
        Generate depth map for each proposal
        Args:
            depth_map: shape(N, 1, H, W)
            proposals: shape(N, M, 4)
        Returns:
            proposals_depth_map: shape(N, M, mask_size*mask_size)
        """
        # import matplotlib.pyplot as plt
        N, M = proposals.shape[:2]
        H, W = depth_map.shape[-2:]
        proposals = proposals.long()
        depth_targets = []
        for batch_ind in range(N):
            proposals_single_image = proposals[batch_ind]
            pos_proposals_single_image = proposals_single_image[
                weights[batch_ind] > 0]
            for proposals_ind in range(pos_proposals_single_image.shape[0]):
                box = pos_proposals_single_image[proposals_ind]
                depth_targets.append(
                    F.upsample_bilinear(
                        depth_map[batch_ind:batch_ind + 1, :, box[1]:box[3],
                                  box[0]:box[2]],
                        size=mask_size))
                # plt.imshow(depth_targets[-1].cpu().numpy()[0, 0])
                # plt.show()

        if len(depth_targets) == 0:
            depth_targets = torch.zeros(1, 0, 28, 28).type_as(depth_map)
        else:
            depth_targets = torch.cat(depth_targets, dim=1)
        return depth_targets.view(-1, mask_size * mask_size)

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

    def init_modules(self):
        super().init_modules()
        # not class agnostic for dims
        self.rcnn_dim_preds = nn.ModuleList(
            [nn.Linear(1024, 3) for _ in range(self.num_stages)])

        self.maskrcnn_pooling = AdaptiveROIAlign((14, 14), 2)
        if self.depth_enable:
            self.depth_map_predictor = KeyPointPredictor(256, 1)
        if self.mask_enable:
            self.instance_map_predictor = KeyPointPredictor(256, 2)
        self.rcnn_depth_preds = nn.ModuleList(
            [nn.Linear(1024, 1) for _ in range(self.num_stages)])
        self.l1_loss = nn.L1Loss(reduction='none')
        self.l2_loss = nn.MSELoss(reduction='none')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')
        self.mask_loss = nn.CrossEntropyLoss(reduction='none')

    def depth_map_loss(self, mask_target, center_depth_gt):
        """
        Args:
            mask_preds: shape(N,M,-1)
            pos_mask: shape(num_pos, -1)
            weights: shape(N,M)
        """
        # import ipdb
        # ipdb.set_trace()
        mask_preds = mask_target['pred']
        pos_mask = mask_target['target']
        weights = mask_target['weight']
        fg_map = mask_target['fg_weight']
        # positive sample filter
        pos_filter = weights.view(-1) > 0
        pos_mask_preds = mask_preds.view(-1, mask_preds.shape[-1])[pos_filter]
        pos_center_depth_gt = center_depth_gt.view(
            -1, center_depth_gt.shape[-1])[pos_filter]

        pos_mask_deltas_gt = pos_mask - pos_center_depth_gt

        # non zero filter
        non_zero_filter = pos_mask > 0
        non_zero_mask_deltas_preds = pos_mask_preds[non_zero_filter]
        non_zero_mask_deltas_gt = pos_mask_deltas_gt[non_zero_filter]
        non_zero_fg_map = fg_map[non_zero_filter]
        # non_zero_center_depth_gt = pos_center_depth_gt[non_zero_filter]

        # fg mask filter

        # import ipdb
        # ipdb.set_trace()
        # prevent bg and ground plane
        fg_filter = non_zero_fg_map > 0
        fg_filter = (torch.abs(non_zero_mask_deltas_gt) <
                     2.5) & (non_zero_fg_map > 0)

        mask_loss = self.smooth_l1_loss(non_zero_mask_deltas_preds[fg_filter],
                                        non_zero_mask_deltas_gt[fg_filter])
        # num_pos = weights[weights > 0].float().sum()
        # num_pos = num_pos.clamp(min=1)
        return mask_loss

    def instance_mask_loss(self, mask_target):
        # import ipdb
        # ipdb.set_trace()
        mask_preds = mask_target['pred']
        pos_mask = mask_target['target']
        weights = mask_target['weight']

        # positive sample filter
        pos_filter = weights.view(-1) > 0
        pos_mask_preds = mask_preds.view(-1, mask_preds.shape[-1])[pos_filter]
        if pos_mask.numel() == 0:
            return torch.zeros(1).type_as(pos_mask_preds)
        mask_loss = self.mask_loss(
            pos_mask_preds.view(-1, self.n_classes), pos_mask.view(-1))
        return mask_loss

    def loss(self, prediction_dict, feed_dict):
        """
        assign proposals label and subsample from them
        Then calculate loss
        """
        # import ipdb
        # ipdb.set_trace()
        loss_dict = super().loss(prediction_dict, feed_dict)
        targets = prediction_dict[constants.KEY_TARGETS]
        # center_depth = prediction_dict['center_depth']
        # rcnn_corners_loss = 0
        rcnn_mask_loss = torch.zeros(1).float().to('cuda')
        rcnn_depth_loss = torch.zeros(1).float().to('cuda')
        rcnn_instance_loss = torch.zeros(1).float().to('cuda')
        # rcnn_dim_loss = 0

        for stage_ind in range(self.num_stages):
            # rcnn_corners_loss = rcnn_corners_loss + common_loss.calc_loss(
            # self.rcnn_cls_loss, orient_target, True)

            # import ipdb
            # ipdb.set_trace()

            if self.mask_enable:
                instance_mask_target = targets[stage_ind][-1]
                weights = instance_mask_target['weight']
                num_pos = weights[weights > 0].float().sum()
                if num_pos > 0:
                    # num_pos = num_pos.clamp(min=1)
                    rcnn_instance_loss = rcnn_instance_loss + self.instance_mask_loss(
                        instance_mask_target).sum() / num_pos

            if self.depth_enable:
                depth_target = targets[stage_ind][3]
                center_depth_preds = depth_target['pred']
                center_depth_gt = depth_target['target'][:, :, 26:27]
                weights = depth_target['weight']
                num_pos = weights[weights > 0].float().sum()
                if num_pos > 0:
                    # num_pos = num_pos.clamp(min=1)

                    rcnn_depth_loss = rcnn_depth_loss + (
                        self.l1_loss(center_depth_preds, center_depth_gt
                                     ) * weights.unsqueeze(-1)).sum() / num_pos

                    depth_map_target = targets[stage_ind][2]
                    rcnn_mask_loss = rcnn_mask_loss + self.depth_map_loss(
                        depth_map_target, center_depth_gt).sum() / num_pos
            # dim_target = targets[stage_ind][3]
            # rcnn_dim_loss = rcnn_dim_loss + common_loss.calc_loss(
            # self.rcnn_bbox_loss, dim_target, True)

            # instance mask loss

        if self.mask_enable:
            loss_dict.update({'rcnn_instance_loss': rcnn_instance_loss})

        if self.depth_enable:
            loss_dict.update({
                'rcnn_mask_loss': rcnn_mask_loss,
                'rcnn_depth_loss': rcnn_depth_loss
            })

        return loss_dict
