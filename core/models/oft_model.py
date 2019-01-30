# -*- coding: utf-8 -*-
"""
use one stage detector as the framework to detect 3d object
in OFT feature map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.model import Model
from core.models.feature_extractors.oft import OFTNetFeatureExtractor
from core.voxel_generator import VoxelGenerator
from core.oft_target_assigner import TargetAssigner
from core.models.focal_loss import FocalLoss
from core.samplers.detection_sampler import DetectionSampler
from utils.integral_map import IntegralMapGenerator


class OFTModel(Model):
    def forward(self, feed_dict):
        self.voxel_generator.proj_voxels_3dTo2d(feed_dict['p2'])
        # import ipdb
        # ipdb.set_trace()
        img_feat_maps = self.feature_extractor.forward(feed_dict['img'])

        img_feat_maps = self.feature_preprocess(img_feat_maps)

        integral_maps = self.generate_integral_maps(img_feat_maps)

        # import ipdb
        # ipdb.set_trace()
        oft_maps = self.generate_oft_maps(integral_maps)

        bev_feat_maps = self.feature_extractor.bev_feature(oft_maps)

        # pred output
        # shape (NCHW)
        output_maps = self.output_head(bev_feat_maps)
        # shape(N,M,out_channels)
        pred_3d = output_maps.permute(0, 2, 3, 1).contiguous().view(
            self.batch_size, -1, self.output_channels)

        pred_boxes_3d = pred_3d[:, :, self.n_classes:]
        pred_scores_3d = pred_3d[:, :, :self.n_classes]

        pred_probs_3d = F.softmax(pred_scores_3d, dim=-1)

        prediction_dict = {}
        prediction_dict['pred_boxes_3d'] = pred_boxes_3d
        prediction_dict['pred_scores_3d'] = pred_scores_3d
        prediction_dict['pred_probs_3d'] = pred_probs_3d
        return prediction_dict

    def feature_preprocess(self, feat_maps):
        # import ipdb
        # ipdb.set_trace()
        reduced_feat_maps = []
        for ind, feat_map in enumerate(feat_maps):
            reduced_feat_map = self.feats_reduces[ind](feat_map)
            reduced_feat_maps.append(reduced_feat_map)
        return reduced_feat_maps

    def generate_integral_maps(self, img_feat_maps):
        integral_maps = []
        for img_feat_map in img_feat_maps:
            integral_maps.append(
                self.integral_map_generator.generate(img_feat_map))

        return integral_maps

    def generate_oft_maps(self, integral_maps):
        # shape(N,4)
        voxel_proj_2d = self.voxel_generator.voxel_proj_2d
        # for i in range(voxel_proj_2d.shape[0]):
        multiscale_img_feat = []
        for integral_map in integral_maps:
            multiscale_img_feat.append(
                self.integral_map_generator.calc(integral_map, voxel_proj_2d))

        # shape(N,C,HWD)
        fusion_feat = multiscale_img_feat[0] + multiscale_img_feat[
            1] + multiscale_img_feat[2]
        depth_dim = self.voxel_generator.lattice_dims[1]
        height_dim = self.voxel_generator.lattice_dims[0]

        fusion_feat = fusion_feat.view(
            self.batch_size, self.feat_size, -1,
            depth_dim).permute(0, 3, 1, 2).contiguous()
        # shape(N,C,HW)
        oft_maps = self.feat_collapse(fusion_feat).view(
            self.batch_size, self.feat_size, height_dim, -1)

        return oft_maps

    def init_param(self, model_config):

        self.feat_size = model_config['common_feat_size']
        self.batch_size = model_config['batch_size']
        self.sample_size = model_config['sample_size']
        self.n_classes = model_config['num_classes']
        self.use_focal_loss = model_config['use_focal_loss']
        self.feature_extractor_config = model_config['feature_extractor_config']

        self.voxel_generator = VoxelGenerator(
            model_config['voxel_generator_config'])
        self.voxel_generator.init_voxels()

        self.integral_map_generator = IntegralMapGenerator()

        self.target_assigner = TargetAssigner(
            model_config['target_assigner_config'])

        self.sampler = DetectionSampler(model_config['sampler_config'])

        self.reg_channels = 3 + 3 + 2

        # score, pos, dim, ang
        self.output_channels = self.n_classes + self.reg_channels

    def init_modules(self):
        """
        some modules
        """

        self.feature_extractor = OFTNetFeatureExtractor(
            self.feature_extractor_config)

        feats_reduce_1 = nn.Conv2d(128, self.feat_size, 1, 1, 0)
        feats_reduce_2 = nn.Conv2d(256, self.feat_size, 1, 1, 0)
        feats_reduce_3 = nn.Conv2d(512, self.feat_size, 1, 1, 0)
        self.feats_reduces = nn.ModuleList(
            [feats_reduce_1, feats_reduce_2, feats_reduce_3])

        self.feat_collapse = nn.Conv2d(8, 1, 1, 1, 0)

        self.output_head = nn.Conv2d(256 * 4, self.output_channels, 1, 1, 0)

        # loss
        self.reg_loss = nn.L1Loss(reduce=False)
        # self.reg_loss = nn.SmoothL1Loss(reduce=False)
        if self.use_focal_loss:
            self.conf_loss = FocalLoss(
                self.n_classes, alpha=0.2, gamma=2, auto_alpha=False)
        else:
            self.conf_loss = nn.CrossEntropyLoss(reduce=False)

    def init_weights(self):
        self.feature_extractor.init_weights()

    def loss(self, prediction_dict, feed_dict):
        # import ipdb
        # ipdb.set_trace()
        gt_boxes_3d = feed_dict['gt_boxes_3d']
        gt_labels = feed_dict['gt_labels']
        gt_boxes_ground_2d_rect = feed_dict['gt_boxes_ground_2d_rect']

        voxels_ground_2d = self.voxel_generator.proj_voxels_to_ground()
        voxel_centers = self.voxel_generator.voxel_centers
        D = self.voxel_generator.lattice_dims[1]
        voxel_centers = voxel_centers.view(-1, D, 3)[:, 0, :]

        # gt_boxes_3d = torch.cat([gt_boxes_3d[:,:,:3],],dim=-1)

        cls_weights, reg_weights, cls_targets, reg_targets = self.target_assigner.assign(
            voxels_ground_2d, gt_boxes_ground_2d_rect, voxel_centers,
            gt_boxes_3d, gt_labels)

        # pred_boxes_3d = prediction_dict['pred_boxes_3d']
        ################################
        # subsample
        ################################

        pos_indicator = reg_weights > 0
        indicator = cls_weights > 0

        rpn_cls_probs = prediction_dict['pred_probs_3d'][:, :, 1]
        cls_criterion = rpn_cls_probs

        batch_sampled_mask = self.sampler.subsample_batch(
            self.sample_size,
            pos_indicator,
            criterion=cls_criterion,
            indicator=indicator)
        batch_sampled_mask = batch_sampled_mask.type_as(cls_weights)
        rpn_cls_weights = cls_weights * batch_sampled_mask
        rpn_reg_weights = reg_weights * batch_sampled_mask
        num_cls_coeff = (rpn_cls_weights > 0).sum(dim=1)
        num_reg_coeff = (rpn_reg_weights > 0).sum(dim=1)
        # check
        #  assert num_cls_coeff, 'bug happens'
        #  assert num_reg_coeff, 'bug happens'
        if num_cls_coeff == 0:
            num_cls_coeff = torch.ones([]).type_as(num_cls_coeff)
        if num_reg_coeff == 0:
            num_reg_coeff = torch.ones([]).type_as(num_reg_coeff)

        # cls loss
        rpn_cls_score = prediction_dict['pred_scores_3d']
        rpn_cls_loss = self.conf_loss(
            rpn_cls_score.view(-1, 2), cls_targets.view(-1))
        rpn_cls_loss = rpn_cls_loss.view_as(rpn_cls_weights)
        rpn_cls_loss *= rpn_cls_weights
        rpn_cls_loss = rpn_cls_loss.sum(dim=1) / num_cls_coeff.float()

        # bbox loss
        # shape(N,num,4)

        rpn_bbox_preds = prediction_dict['pred_boxes_3d']
        rpn_reg_loss = self.reg_loss(rpn_bbox_preds, reg_targets)
        rpn_reg_loss *= rpn_reg_weights.unsqueeze(-1).expand(-1, -1,
                                                             self.reg_channels)
        rpn_reg_loss = rpn_reg_loss.view(rpn_reg_loss.shape[0], -1).sum(
            dim=1) / num_reg_coeff.float()

        prediction_dict['rcnn_reg_weights'] = rpn_reg_weights[
            batch_sampled_mask > 0]

        loss_dict = {}

        loss_dict['rpn_cls_loss'] = rpn_cls_loss
        loss_dict['rpn_bbox_loss'] = rpn_reg_loss

        # recall
        # final_boxes = self.bbox_coder.decode_batch(rpn_bbox_preds, )
        # self.target_assigner.assign(final_boxes, gt_boxes)
        return loss_dict
