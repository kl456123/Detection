# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import Model
from core.models.real_avod_rpn_model import RPNModel
from core.models.focal_loss import FocalLoss
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.psroi_pooling.modules.psroi_pool import PSRoIPool

from core.filler import Filler
from core.avod_target_assigner import TargetAssigner
from core.samplers.hard_negative_sampler import HardNegativeSampler
from core.samplers.balanced_sampler import BalancedSampler
from core.models.feature_extractors.resnet import ResNetFeatureExtractor
from core.samplers.detection_sampler import DetectionSampler
from core.models.avod_basic_fc import AVODBasicFC
from core.profiler import Profiler
from lib.model.nms.nms_wrapper import nms
from core.similarity_calc.center_similarity_calc import CenterSimilarityCalc

import functools
from utils import box_ops

from core.avod import anchor_projector
from core.avod import anchor_encoder
from core.avod import box_3d_encoder
from core.avod import box_4c_encoder
from core.avod import orientation_encoder


class RealAVODFasterRCNN(Model):
    def create_img_bbox_filter(self, img_bboxes, gt_boxes_2d):
        match_quality_matrix = self.iou_calc.compare_batch(img_bboxes,
                                                           gt_boxes_2d)
        max_overlaps, argmax_overlaps = torch.max(match_quality_matrix, dim=2)

        max_overlaps = max_overlaps

        _, order = torch.sort(max_overlaps, descending=True, dim=-1)
        img_bbox_filter = max_overlaps > 0.1
        idx = torch.nonzero(img_bbox_filter)
        if idx.numel() == 0:
            return torch.ones_like(img_bbox_filter).type_as(img_bbox_filter)
        else:
            return img_bbox_filter

        return img_bbox_filter

    def anchors_norm2_anchors(self, bev_anchors_norm, bev_shape):
        extents_tiled = [bev_shape[::-1], bev_shape[::-1]]
        bev_anchors = bev_anchors_norm * torch.tensor(extents_tiled).view(
            -1, 4).type_as(bev_anchors_norm)
        return bev_anchors

    def forward(self, feed_dict):

        prediction_dict = {}

        # rpn model
        prediction_dict.update(self.rpn_model.forward(feed_dict))

        # rois_batch = prediction_dict['rois_batch']
        top_proposals = prediction_dict['proposals_batch']
        ground_plane = feed_dict['ground_plane']
        class_labels = feed_dict['label_classes']

        # expand rois
        expand_length = self.expand_proposals_xz
        if expand_length > 0:
            expanded_dim_x = top_proposals[:, :, 3] + expand_length
            expanded_dim_z = top_proposals[:, :, 5] + expand_length
            expanded_rois_batch = torch.stack(
                [
                    top_proposals[:, :, 0], top_proposals[:, :, 1],
                    top_proposals[:, :, 2], expanded_dim_x,
                    top_proposals[:, :, 4], expanded_dim_z
                ],
                dim=-1)
            top_proposals = expanded_rois_batch

        # avod_projection
        # bev

        bev_feat_maps = feed_dict['bev_feat_maps']
        img_feat_maps = feed_dict['img_feat_maps']
        # import ipdb
        # ipdb.set_trace()

        # img
        image_shape = img_feat_maps.shape[-2:]
        img_proposal_boxes, img_proposal_boxes_norm = anchor_projector.torch_project_to_image_space(
            top_proposals, feed_dict['stereo_calib_p2'][0], image_shape)

        # use img filter again
        label_anchors = feed_dict['label_anchors']
        img_anchors_gt, img_anchors_gt_norm = anchor_projector.torch_project_to_image_space(
            label_anchors[0], feed_dict['stereo_calib_p2'][0], image_shape)
        img_bbox_filter = self.create_img_bbox_filter(
            img_proposal_boxes.unsqueeze(0), img_anchors_gt.unsqueeze(0))[0]

        top_proposals = top_proposals[img_bbox_filter]
        img_proposal_boxes = img_proposal_boxes[img_bbox_filter]

        bev_proposal_boxes, bev_proposal_boxes_norm = anchor_projector.project_to_bev(
            top_proposals, self.bev_extents)
        bev_shape = bev_feat_maps.shape[-2:]
        extents_tiled = [bev_shape[::-1], bev_shape[::-1]]
        bev_proposal_boxes_pixel = bev_proposal_boxes_norm * torch.tensor(
            extents_tiled).view(-1, 4).type_as(bev_proposal_boxes_norm)
        roi_indexes = torch.zeros_like(bev_proposal_boxes_pixel[:, -1:])
        bev_rois_boxes = torch.cat([roi_indexes, bev_proposal_boxes_pixel],
                                   dim=-1)

        img_rois_boxes = torch.cat([roi_indexes, img_proposal_boxes], dim=-1)

        # path drop
        if not (self.path_drop_probabilities[0] ==
                self.path_drop_probabilities[1] == 1.0):
            img_mask = prediction_dict['img_mask']
            bev_mask = prediction_dict['bev_mask']
            img_feat_maps = img_mask * img_feat_maps
            bev_feat_maps = bev_mask * bev_feat_maps
        else:
            bev_mask = 1.0
            img_mask = 1.0

        bev_rois = self.rcnn_pooling(bev_feat_maps, bev_rois_boxes)
        img_rois = self.rcnn_pooling(img_feat_maps, img_rois_boxes)

        # output
        all_cls_logits, all_offsets, all_angle_vectors = self.fc_output_layer(
            [img_rois, bev_rois], [bev_mask, img_mask])

        all_cls_softmax = F.softmax(all_cls_logits, dim=-1)

        # combine all_offsets and angles
        all_reg_3d = torch.cat([all_offsets, all_angle_vectors], dim=-1)

        # prepare something for subsample

        prediction_dict['all_cls_logits'] = all_cls_logits
        # prediction_dict['all_offsets'] = all_offsets
        # prediction_dict['all_angle_vectors'] = all_angle_vectors
        prediction_dict['all_reg_3d'] = all_reg_3d
        prediction_dict['all_cls_softmax'] = all_cls_softmax

        # used for matching
        # prediction_dict['bev_proposal_boxes'] = bev_proposal_boxes
        prediction_dict['proposals_batch'] = top_proposals

        proposal_boxes_3d = box_3d_encoder.anchors_to_box_3d(
            top_proposals, fix_lw=True)
        proposal_boxes_4c = box_4c_encoder.torch_box_3d_to_box_4c(
            proposal_boxes_3d, ground_plane[0])
        prediction_dict['proposal_boxes_4c'] = proposal_boxes_4c

        # when inference
        if not self.training:
            all_orientations = orientation_encoder.torch_angle_vector_to_orientation(
                all_angle_vectors)
            prediction_boxes_4c = box_4c_encoder.torch_offsets_to_box_4c(
                proposal_boxes_4c, all_offsets)
            prediction_boxes_3d = box_4c_encoder.torch_box_4c_to_box_3d(
                prediction_boxes_4c, ground_plane[0])
            prediction_anchors = box_3d_encoder.torch_box_3d_to_anchor(
                prediction_boxes_3d)
            # nms for final detection
            avod_bev_boxes, avod_bev_boxes_norm = anchor_projector.project_to_bev(
                prediction_anchors, self.area_extents)
            avod_bev_boxes_pixel = self.anchors_norm2_anchors(
                avod_bev_boxes_norm, bev_shape)
            avod_bev_boxes_pixel = torch.round(avod_bev_boxes_pixel)

            # keep_idx, _ = box_ops.nms(avod_bev_boxes_pixel,
            # all_cls_softmax[:, -1], self.nms_thresh)
            # keep_idx = keep_idx.long().view(-1)
            # final_bboxes_3d = prediction_boxes_3d[keep_idx]
            # all_cls_softmax = all_cls_softmax[keep_idx]
            # all_orientations= all_orientations[keep_idx]

            # import ipdb
            # ipdb.set_trace()
            # final_bboxes_3d = torch.cat(
            # [final_bboxes_3d, all_orientations.unsqueeze(0)], dim=-1)
            prediction_dict['final_pred_boxes_3d'] = prediction_boxes_3d
            prediction_dict['final_pred_orientations'] = all_orientations
            prediction_dict['all_cls_softmax'] = all_cls_softmax

        return prediction_dict

    def init_weights(self):
        # submodule init weights
        # self.feature_extractor.init_weights()
        self.rpn_model.init_weights()
        self.fc_output_layer.init_weights()

        # Filler.normal_init(self.rcnn_cls_pred, 0, 0.01, self.truncated)
        # Filler.normal_init(self.rcnn_bbox_pred, 0, 0.001, self.truncated)

    def init_modules(self):
        # self.feature_extractor = ResNetFeatureExtractor(
        # self.feature_extractor_config)
        self.rpn_model = RPNModel(self.rpn_config)
        if self.pooling_mode == 'align':
            self.rcnn_pooling = RoIAlignAvg(self.pooling_size,
                                            self.pooling_size, 1.0)
        elif self.pooling_mode == 'ps':
            self.rcnn_pooling = PSRoIPool(7, 7, 1.0 / 16, 7, self.n_classes)
        elif self.pooling_mode == 'psalign':
            raise NotImplementedError('have not implemented yet!')
        elif self.pooling_mode == 'deformable_psalign':
            raise NotImplementedError('have not implemented yet!')

        # loss module
        if self.use_focal_loss:
            self.rcnn_cls_loss = FocalLoss(
                self.n_classes, alpha=0.25, gamma=2, auto_alpha=False)
        else:
            self.rcnn_cls_loss = nn.CrossEntropyLoss(reduce=False)
            # self.rcnn_cls_loss = functools.partial(
        # F.cross_entropy, reduce=False)

        self.rcnn_bbox_loss = nn.modules.SmoothL1Loss(reduce=False)

        self.fc_output_layer = AVODBasicFC(self.fc_output_layer_config)
        # self.fc_output_layer = AvodFusionModel()

    def init_param(self, model_config):
        self.iou_calc = CenterSimilarityCalc()
        self.nms_thresh = model_config['nms_thresh']
        self.path_drop_probabilities = model_config['path_drop_probabilities']
        self.area_extents = model_config['area_extents']
        self.bev_extents = [self.area_extents[0], self.area_extents[2]]

        classes = model_config['classes']
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = model_config['class_agnostic']
        self.pooling_size = model_config['pooling_size']
        self.pooling_mode = model_config['pooling_mode']
        self.truncated = model_config['truncated']
        self.expand_proposals_xz = model_config['expand_proposals_xz']

        self.use_focal_loss = model_config['use_focal_loss']
        self.rcnn_batch_size = model_config['rcnn_batch_size']

        # some submodule config
        self.feature_extractor_config = model_config['feature_extractor_config']
        self.rpn_config = model_config['rpn_config']
        self.fc_output_layer_config = model_config['fc_output_layer_config']

        # assigner
        self.target_assigner = TargetAssigner(
            model_config['target_assigner_config'])

        # coder
        self.bbox_coder = self.target_assigner.bbox_coder

        # sampler
        self.sampler = BalancedSampler(model_config['sampler_config'])

        self.reduce = True

        self.profiler = Profiler()

    def loss(self, prediction_dict, feed_dict):
        # loss for cls

        loss_dict = {}
        loss_dict.update(self.rpn_model.loss(prediction_dict, feed_dict))
        gt_boxes_3d = feed_dict['label_boxes_3d']
        top_proposals = prediction_dict['proposals_batch'].unsqueeze(0)

        # import ipdb
        # ipdb.set_trace()
        # increate batch dim
        gt_boxes_bev, gt_boxes_bev_norm = anchor_projector.project_to_bev(
            gt_boxes_3d[0], self.bev_extents)
        top_boxes_bev, top_boxes_bev_norm = anchor_projector.project_to_bev(
            top_proposals[0], self.bev_extents)
        bev_shape = feed_dict['bev_feat_maps'].shape[-2:]

        gt_boxes_bev_pixel = self.anchors_norm2_anchors(gt_boxes_bev_norm,
                                                        bev_shape)
        boxes_bev_pixel = self.anchors_norm2_anchors(top_boxes_bev_norm,
                                                     bev_shape)

        gt_boxes_bev_pixel = torch.round(gt_boxes_bev_pixel).unsqueeze(0)
        boxes_bev_pixel = torch.round(boxes_bev_pixel).unsqueeze(0)
        proposal_boxes_4c = prediction_dict['proposal_boxes_4c'].unsqueeze(0)

        #################################
        # target assigner
        ################################
        rcnn_cls_targets, rcnn_reg_targets, \
            rcnn_cls_weights, rcnn_reg_weights = \
                self.target_assigner.assign(
                    boxes_bev_pixel,
                    gt_boxes_bev_pixel,
                    proposal_boxes_4c,
                    gt_boxes_3d,
                    gt_labels=None,
                    ground_plane=feed_dict['ground_plane'])

        ################################
        # subsample
        ################################
        pos_indicator = rcnn_reg_weights > 0
        indicator = rcnn_cls_weights > 0

        rcnn_cls_probs = prediction_dict['all_cls_softmax'][:, 1]
        cls_criterion = rcnn_cls_probs

        batch_sampled_mask = self.sampler.subsample_batch(
            self.rcnn_batch_size,
            pos_indicator,
            criterion=cls_criterion,
            indicator=indicator)
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
        rcnn_cls_score = prediction_dict['all_cls_logits']
        rcnn_cls_loss = self.rcnn_cls_loss(
            rcnn_cls_score.view(-1, 2), rcnn_cls_targets.view(-1))
        rcnn_cls_loss = rcnn_cls_loss.view_as(rcnn_cls_weights)
        rcnn_cls_loss = rcnn_cls_loss * rcnn_cls_weights
        rcnn_cls_loss = rcnn_cls_loss.sum(dim=1) / num_cls_coeff.float()

        # bbox loss
        rcnn_bbox_preds = prediction_dict['all_reg_3d'].unsqueeze(0)
        rcnn_reg_loss = self.rcnn_bbox_loss(rcnn_bbox_preds, rcnn_reg_targets)
        rcnn_reg_loss = rcnn_reg_loss * rcnn_reg_weights.unsqueeze(-1)
        rcnn_reg_loss = rcnn_reg_loss.view(rcnn_reg_loss.shape[0], -1).sum(
            dim=1) / num_reg_coeff.float()

        loss_dict['rcnn_cls_loss'] = rcnn_cls_loss * 10
        loss_dict['rcnn_bbox_loss'] = rcnn_reg_loss

        prediction_dict['rcnn_reg_weights'] = rcnn_reg_weights[
            batch_sampled_mask.byte()]

        fake_match = self.target_assigner.analyzer.match
        num_gt = feed_dict['label_classes'].numel()
        self.target_assigner.analyzer.analyze_ap(
            fake_match, rcnn_cls_probs.detach(), num_gt, thresh=0.5)
        return loss_dict
