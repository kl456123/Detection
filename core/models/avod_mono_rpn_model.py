# -*- coding: utf-8 -*-
"""
use one stage detector as the framework to detect 3d object
in OFT feature map
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.model import Model
# from core.models.feature_extractors.avod_vgg_pyramid import AVODVGGPyramidExtractor
from core.models.feature_extractors.vgg_fpn import VGGFPN
from core.voxel_generator import VoxelGenerator
from core.avod_target_assigner import TargetAssigner
from core.models.focal_loss import FocalLoss
from core.samplers.balanced_sampler import BalancedSampler
from core.profiler import Profiler
from core.anchor_projector import AnchorProjector
from core.models.feature_extractors.pvanet import ConvBnAct
from model.roi_align.modules.roi_align import RoIAlignAvg
from core.models.anchor_predictor import AnchorPredictor
from lib.model.nms.nms_wrapper import nms
from core.anchor_generators.grid_anchor_3d_generator import GridAnchor3dGenerator

from utils import box_ops
from utils import pc_ops

from core.similarity_calc.center_similarity_calc import CenterSimilarityCalc
from core.filler import Filler


class RPNModel(Model):
    def forward(self, feed_dict):

        self.preprocess(feed_dict)

        img_feat_maps = feed_dict['img_feat_maps']

        # bottleneck
        img_bottle_neck = self.img_bottle_neck(img_feat_maps)

        if not (self.path_drop_probabilities[0] ==
                self.path_drop_probabilities[1] == 1.0):
            random_values = torch.random()
            img_mask, bev_mask = self._create_path_drop_mask(
                self.path_drop_probabilities[0],
                self.path_drop_probabilities[1], random_values)

            img_bottle_neck = img_mask * img_bottle_neck

        img_proposal_input = img_bottle_neck

        anchors = feed_dict['anchor_boxes_3d_to_use_norm']

        img_anchors_norm = feed_dict['img_norm']

        anchor_indexes = torch.zeros_like(img_anchors_norm[:, :1])
        img_rois_norm = torch.cat([anchor_indexes, img_anchors_norm], dim=-1)

        img_proposal_rois = self.roi_pooling(img_proposal_input, img_rois_norm)

        rpn_fusion_out = img_proposal_rois

        rpn_cls_scores, rpn_bbox_preds = self.anchor_predictor.forward(
            rpn_fusion_out)

        rpn_cls_probs = F.softmax(rpn_cls_scores, dim=-1)

        # 3d nms problem is reduced to 2d nms in bev
        proposals_batch = self.generate_proposal(
            rpn_cls_probs, anchors, rpn_bbox_preds, feed_dict['p2'])
        if self.training:
            gt_boxes_3d = feed_dict['gt_boxes_3d']
            proposals_batch = self.append_gt(proposals_batch, gt_boxes_3d)

        predict_dict = {
            'proposals_batch': proposals_batch,
            'rpn_cls_scores': rpn_cls_scores,

            # used for loss
            'rpn_bbox_preds': rpn_bbox_preds,
            'rpn_cls_probs': rpn_cls_probs,
        }

        return predict_dict

    def create_img_bbox_filter(self, img_bboxes, gt_boxes_2d):
        # import ipdb
        # ipdb.set_trace()
        match_quality_matrix = self.iou_calc.compare_batch(img_bboxes,
                                                           gt_boxes_2d)
        max_overlaps, argmax_overlaps = torch.max(match_quality_matrix, dim=2)

        max_overlaps = max_overlaps.squeeze(0)

        _, order = torch.sort(max_overlaps, descending=True)
        img_bbox_filter = max_overlaps > self.img_filter_iou_thresh
        if self.img_filter_topN > 0:
            img_bbox_filter[order[self.img_filter_topN:]] = 0

        return img_bbox_filter

    def preprocess(self, feed_dict):
        ground_plane = feed_dict['ground_plane']
        all_anchor_boxes_3d = self.anchor_generator.generate_pytorch(
            ground_plane)

        anchor_boxes_3d_to_use = all_anchor_boxes_3d

        img_norm = self.anchor_projector.project_to_image_space(
            anchor_boxes_3d_to_use, feed_dict['p2'])

        gt_boxes_3d = feed_dict['gt_boxes_3d']
        gt_boxes_img = self.anchor_projector.project_to_image_space(
            gt_boxes_3d[0], feed_dict['p2'])

        if self.enable_img_filter:
            img_bbox_filter = self.create_img_bbox_filter(
                img_norm.unsqueeze(0), gt_boxes_img.unsqueeze(0))
        else:
            img_bbox_filter = torch.ones_like(img_norm[:, 0]).byte()

        # area filter(ignore too large bboxes in image)
        img_norm_area = (img_norm[:, 2] - img_norm[:, 0]) * (
            img_norm[:, 3] - img_norm[:, 1])
        img_norm_area_filter = img_norm_area < 1280 * 384 * 0.5

        final_bbox_filter = img_bbox_filter & img_norm_area_filter

        # make sure no empty anchors
        if torch.nonzero(final_bbox_filter).numel() == 0:
            final_bbox_filter[:2000] = 1

        feed_dict['img_norm'] = img_norm[final_bbox_filter]
        feed_dict['anchor_boxes_3d_to_use_norm'] = anchor_boxes_3d_to_use[
            final_bbox_filter]

    def generate_proposal(self, rpn_cls_probs, anchors, rpn_bbox_preds, p2):
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
        rpn_cls_probs = rpn_cls_probs.detach()
        rpn_bbox_preds = rpn_bbox_preds.detach()

        proposals = self.bbox_coder.decode_batch(rpn_bbox_preds, anchors)

        # fg prob
        fg_probs = rpn_cls_probs[:, -1]

        # pre nms
        # sort fg
        _, fg_probs_order = torch.sort(fg_probs, descending=True)
        keep = fg_probs_order
        fg_probs = fg_probs[keep]
        proposals = proposals[keep]

        img_proposals = self.anchor_projector.project_to_image_space(proposals,
                                                                     p2)

        # bev_proposals = self.anchor_projector.project_to_bev(proposals,
        # self.bev_extents)
        # bev_proposals = bev_proposals * 10
        # bev_proposals = torch.round(bev_proposals)

        # nms
        keep_idx = nms(torch.cat((img_proposals, fg_probs.unsqueeze(-1)),
                                 dim=1),
                       self.nms_thresh)
        keep_idx = keep_idx.long().view(-1)
        top_proposals = proposals[keep_idx]

        # post nms
        top_proposals = top_proposals[:self.post_nms_topN]

        return top_proposals

    def _create_path_drop_mask(self):
        return 1, 1

    def init_param(self, model_config):
        self.enable_img_filter = model_config['enable_img_filter']
        self.truncated = model_config['truncated']
        self.img_filter_topN = model_config['img_filter_topN']
        self.use_empty_filter = model_config.get('use_empty_filter')
        self.nms_thresh = model_config['nms_thresh']
        self.img_filter_iou_thresh = model_config['img_filter_iou_thresh']
        self.pre_nms_topN = model_config['pre_nms_topN']
        self.post_nms_topN = model_config['post_nms_topN']
        self.batch_size = model_config['rpn_batch_size']
        self.voxel_size = model_config['voxel_size']
        self.n_classes = model_config['num_classes']
        self.use_focal_loss = model_config['use_focal_loss']
        self.pooling_size = model_config['pooling_size']
        self.bev_feature_extractor_config = model_config[
            'bev_feature_extractor_config']
        self.img_feature_extractor_config = model_config[
            'img_feature_extractor_config']
        self.path_drop_probabilities = model_config['path_drop_probabilities']
        self.fusion_method = model_config['fusion_method']

        self.target_assigner = TargetAssigner(
            model_config['target_assigner_config'])

        self.anchor_predictor_config = model_config['anchor_predictor_config']

        # self.target_assigner.analyzer.append_gt = False

        self.sampler = BalancedSampler(model_config['sampler_config'])

        self.bbox_coder = self.target_assigner.bbox_coder

        self.reg_channels = 3 + 3 + 2

        self.area_extents = model_config['area_extents']
        # just range along xz axis
        self.bev_extents = [self.area_extents[0], self.area_extents[2]]

        self.iou_calc = CenterSimilarityCalc()

        # score, pos, dim, ang
        # self.output_channels = self.n_classes + self.reg_channels

        # find the most expensive operators
        self.profiler = Profiler()

        self.anchor_projector = AnchorProjector()

        self.anchor_generator = GridAnchor3dGenerator(
            model_config['anchor_generator_config'])

    def append_gt(self, rois_batch, gt_boxes):
        ################################
        # append gt_boxes to rois_batch for losses
        ################################

        rois_batch = torch.cat([rois_batch, gt_boxes[0]], dim=0)
        return rois_batch

    def init_modules(self):
        """
        some modules
        """
        self.img_bottle_neck = nn.Sequential(
            nn.Conv2d(
                1024, 1, kernel_size=1), nn.BatchNorm2d(1))

        self.roi_pooling = RoIAlignAvg(self.pooling_size, self.pooling_size,
                                       1.0 / 16.0)
        self.anchor_predictor = AnchorPredictor(self.anchor_predictor_config)

        # loss
        self.rpn_bbox_loss = nn.SmoothL1Loss(reduce=False)
        if self.use_focal_loss:
            self.rpn_cls_loss = FocalLoss(
                self.n_classes, alpha=0.25, gamma=2, auto_alpha=False)
        else:
            self.rpn_cls_loss = nn.CrossEntropyLoss(reduce=False)

    def init_weights(self):
        pass

    def loss(self, prediction_dict, feed_dict):

        loss_dict = {}
        gt_boxes_3d = feed_dict['gt_boxes_3d']
        anchors = feed_dict['anchor_boxes_3d_to_use_norm'].unsqueeze(0)

        # increate batch dim
        gt_boxes_bev = self.anchor_projector.project_to_bev(
            gt_boxes_3d[0], self.bev_extents).unsqueeze(0)
        bev_proposal_boxes_norm = self.anchor_projector.project_to_bev(
            anchors[0], self.bev_extents).unsqueeze(0)

        #################################
        # target assigner
        ################################
        rcnn_cls_targets, rcnn_reg_targets, \
            rcnn_cls_weights, rcnn_reg_weights = \
                self.target_assigner.assign(
                    bev_proposal_boxes_norm,
                    gt_boxes_bev,
                    anchors,
                    gt_boxes_3d,
                    gt_labels=None)

        ################################
        # subsample
        ################################
        pos_indicator = rcnn_reg_weights > 0
        indicator = rcnn_cls_weights > 0

        # rcnn_cls_probs = prediction_dict['rpn_cls_probs'][:, 1]
        # cls_criterion = rcnn_cls_probs

        batch_sampled_mask = self.sampler.subsample_batch(
            self.batch_size, pos_indicator, indicator=indicator)
        batch_sampled_mask = batch_sampled_mask.type_as(rcnn_cls_weights)
        rcnn_cls_weights = rcnn_cls_weights * batch_sampled_mask
        rcnn_reg_weights = rcnn_reg_weights * batch_sampled_mask
        num_cls_coeff = (rcnn_cls_weights > 0).sum(dim=1)
        num_reg_coeff = (rcnn_reg_weights > 0).sum(dim=1)
        # check
        #  assert num_cls_coeff, 'bug happens'
        #  assert num_reg_coeff, 'bug happens'
        if num_cls_coeff == 0:
            num_cls_coeff = torch.ones([]).type_as(num_cls_coeff)
        if num_reg_coeff == 0:
            num_reg_coeff = torch.ones([]).type_as(num_reg_coeff)

        # cls loss
        rcnn_cls_score = prediction_dict['rpn_cls_scores']
        rcnn_cls_loss = self.rpn_cls_loss(
            rcnn_cls_score.view(-1, 2), rcnn_cls_targets.view(-1))
        rcnn_cls_loss = rcnn_cls_loss.view_as(rcnn_cls_weights)
        rcnn_cls_loss = rcnn_cls_loss * rcnn_cls_weights
        rcnn_cls_loss = rcnn_cls_loss.sum(dim=1) / num_cls_coeff.float()

        # bbox loss
        rcnn_bbox_preds = prediction_dict['rpn_bbox_preds'].unsqueeze(0)
        rcnn_reg_loss = self.rpn_bbox_loss(rcnn_bbox_preds,
                                           rcnn_reg_targets[:, :, :-2])
        rcnn_reg_loss = rcnn_reg_loss * rcnn_reg_weights.unsqueeze(-1)
        rcnn_reg_loss = rcnn_reg_loss.view(rcnn_reg_loss.shape[0], -1).sum(
            dim=1) / num_reg_coeff.float()

        loss_dict['rpn_cls_loss'] = rcnn_cls_loss
        loss_dict['rpn_bbox_loss'] = rcnn_reg_loss

        return loss_dict
