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
from core.avod_rpn_target_assigner import TargetAssigner
# from core.models.focal_loss import FocalLoss
from utils.focal_loss import FocalLoss
from core.samplers.balanced_sampler import BalancedSampler
from core.profiler import Profiler
from core.models.feature_extractors.pvanet import ConvBnAct
from model.roi_align.modules.roi_align import RoIAlignAvg
from core.models.anchor_predictor import AnchorPredictor
from lib.model.nms.nms_wrapper import nms
from core.anchor_generators.grid_anchor_3d_generator import GridAnchor3dGenerator

from utils import box_ops

from core.similarity_calc.center_similarity_calc import CenterSimilarityCalc
from core.filler import Filler
from core.avod import anchor_encoder
from core.avod import anchor_projector


class RPNModel(Model):
    def anchors_norm2_anchors(self, bev_anchors_norm, bev_shape):
        extents_tiled = [
            bev_shape[1], bev_shape[0], bev_shape[1], bev_shape[0]
        ]
        try:
            bev_anchors = bev_anchors_norm * torch.tensor(extents_tiled).view(
                -1, 4).type_as(bev_anchors_norm)
        except:
            import ipdb
            ipdb.set_trace()
        return bev_anchors

    def forward(self, feed_dict):
        # import ipdb
        # ipdb.set_trace()
        self.preprocess(feed_dict)

        img_feat_maps = self.img_feature_extractor.forward(feed_dict['img'])
        bev_feat_maps = self.bev_feature_extractor.forward(
            feed_dict['bev_input'])
        # feed_dict['bev_feat_maps'] = bev_feat_maps
        # feed_dict['img_feat_maps'] = img_feat_maps

        # bottleneck
        img_bottle_neck = self.img_bottle_neck(img_feat_maps)
        bev_bottle_neck = self.bev_bottle_neck(bev_feat_maps)

        img_proposal_input = img_bottle_neck
        bev_proposal_input = bev_bottle_neck

        anchors = feed_dict['anchors']
        # print('rpn anchors: ', anchors.shape[0])

        img_anchors_norm = feed_dict['img_anchors_norm']
        bev_anchors_norm = feed_dict['bev_anchors_norm']
        # self.bev_shape = bev_proposal_input.shape[-2:]
        # bev_shape = feed_dict['bev_shape'][0]

        bev_anchors = self.anchors_norm2_anchors(bev_anchors_norm,
                                                 self.bev_shape)
        img_anchors = self.anchors_norm2_anchors(img_anchors_norm,
                                                 self.img_shape)
        anchor_indexes = torch.zeros_like(img_anchors[:, :1])
        img_rois = torch.cat([anchor_indexes, img_anchors], dim=-1)
        bev_rois = torch.cat([anchor_indexes, bev_anchors], dim=-1)

        img_proposal_rois = self.roi_pooling(img_proposal_input, img_rois)
        bev_proposal_rois = self.roi_pooling(bev_proposal_input, bev_rois)

        rpn_fusion_out = torch.cat([bev_proposal_rois, img_proposal_rois],
                                   dim=1)

        rpn_cls_scores, rpn_bbox_preds = self.anchor_predictor.forward(
            rpn_fusion_out)

        rpn_cls_probs = F.softmax(rpn_cls_scores, dim=-1)

        # 3d nms problem is reduced to 2d nms in bev
        proposals_batch = self.generate_proposal(
            rpn_cls_probs, anchors, rpn_bbox_preds, self.bev_shape)
        # if self.training:
        # label_anchors = feed_dict['label_anchors']
        # proposals_batch = self.append_gt(proposals_batch, label_anchors)

        predict_dict = {
            'proposals_batch': proposals_batch,
            'rpn_cls_scores': rpn_cls_scores,

            # used for loss
            'rpn_bbox_preds': rpn_bbox_preds,
            'rpn_cls_probs': rpn_cls_probs,
            'bev_bboxes_norm': bev_anchors_norm,
            'img_feat_maps': img_feat_maps,
            'bev_feat_maps': bev_feat_maps
        }

        return predict_dict

    def create_img_bbox_filter(self, img_bboxes, gt_boxes_2d):
        match_quality_matrix = self.iou_calc.compare_batch(img_bboxes,
                                                           gt_boxes_2d)
        max_overlaps, argmax_overlaps = torch.max(match_quality_matrix, dim=2)

        _, order = torch.sort(max_overlaps, descending=True, dim=-1)
        img_bbox_filter = max_overlaps > self.img_filter_iou_thresh
        if self.img_filter_topN > 0:
            img_bbox_filter[0][order[0][self.img_filter_topN:]] = 0

        return img_bbox_filter

    def preprocess(self, feed_dict):
        # import ipdb
        # ipdb.set_trace()
        img_anchors_norm = feed_dict['img_anchors_norm']
        img_anchors_gt_norm = feed_dict['img_anchors_gt_norm']
        img_anchors = self.anchors_norm2_anchors(img_anchors_norm,
                                                 self.img_shape)
        img_anchors_gt = self.anchors_norm2_anchors(img_anchors_gt_norm,
                                                    self.img_shape)

        # area filter(ignore too large bboxes in image)
        img_norm_area = (img_anchors[:, :, 2] - img_anchors[:, :, 0]) * (
            img_anchors[:, :, 3] - img_anchors[:, :, 1])
        img_norm_area_filter = img_norm_area > 0

        if self.use_img_filter:
            # img_bbox filter
            img_bbox_filter = self.create_img_bbox_filter(img_anchors,
                                                          img_anchors_gt)
        else:
            img_bbox_filter = torch.ones_like(img_norm_area_filter).type_as(
                img_norm_area_filter)

        final_bbox_filter = img_bbox_filter & img_norm_area_filter

        # import ipdb
        # ipdb.set_trace()
        # make sure no empty anchors
        # num_remain = final_bbox_filter.shape[-1]
        num_remain = torch.nonzero(final_bbox_filter[0]).numel()
        if num_remain == 0:
            final_bbox_filter[0, :2000] = 1

        # to prevent out of memory
        # if self.img_filter_topN > 0:
        # final_bbox_filter[0, self.img_filter_topN:] = 0

        # import ipdb
        # ipdb.set_trace()
        # final_bbox_filter[0, 300:] = 0
        # final_bbox_filter[0, :300] = 1

        # bev_anchors = feed_dict['bev_anchors']
        bev_anchors_norm = feed_dict['bev_anchors_norm']
        anchors = feed_dict['anchors']

        feed_dict['anchors'] = anchors[final_bbox_filter]
        feed_dict['bev_anchors_norm'] = bev_anchors_norm[final_bbox_filter]
        # feed_dict['bev_anchors'] = bev_anchors[final_bbox_filter]
        feed_dict['img_anchors_norm'] = img_anchors_norm[final_bbox_filter]

    def generate_proposal(self, rpn_cls_probs, anchors, rpn_bbox_preds,
                          bev_shape):
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

        # proposals = self.bbox_coder.decode_batch(rpn_bbox_preds, anchors)
        regressed_anchors = anchor_encoder.offset_to_anchor(anchors,
                                                            rpn_bbox_preds)

        # pre nms
        fg_probs = rpn_cls_probs[:, -1]

        _, fg_probs_order = torch.sort(fg_probs, descending=True)
        keep = fg_probs_order
        fg_probs = fg_probs[keep]
        regressed_anchors = regressed_anchors[keep]

        _, bev_proposal_boxes_norm = anchor_projector.project_to_bev(
            regressed_anchors, self.bev_extents)

        bev_proposal_boxes_norm = torch.tensor(
            bev_proposal_boxes_norm).type_as(regressed_anchors)

        bev_proposal_boxes = self.anchors_norm2_anchors(
            bev_proposal_boxes_norm, self.bev_shape)
        bev_proposal_boxes = torch.round(bev_proposal_boxes)

        # nms
        keep_idx = nms(torch.cat((bev_proposal_boxes, fg_probs.unsqueeze(-1)),
                                 dim=1),
                       self.nms_thresh)
        keep_idx = keep_idx.long().view(-1)

        top_proposals = regressed_anchors[keep_idx]

        # post nms
        top_proposals = top_proposals[:self.post_nms_topN]

        return top_proposals

    def _create_path_drop_mask(self):
        return 1, 1

    def init_param(self, model_config):
        self.bev_shape = [704, 800]
        self.img_shape = [360, 1200]
        self.use_img_filter = model_config['use_img_filter']
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

        # self.bev_feature_extractor = AVODVGGPyramidExtractor(
        # self.bev_feature_extractor_config)

        # self.img_feature_extractor = AVODVGGPyramidExtractor(
        # self.img_feature_extractor_config)

        # self.bev_bottle_neck = ConvBnAct(
        # 32, 1, kernel_size=1, stride=1, padding=0)
        # self.img_bottle_neck = ConvBnAct(
        # 32, 1, kernel_size=1, stride=1, padding=0)
        self.img_feature_extractor = VGGFPN(in_channels=3)
        self.bev_feature_extractor = VGGFPN(in_channels=6)

        self.bev_bottle_neck = nn.Sequential(
            nn.Conv2d(
                32, 1, kernel_size=1),
            nn.BatchNorm2d(1), )
        self.img_bottle_neck = nn.Sequential(
            nn.Conv2d(
                32, 1, kernel_size=1), nn.BatchNorm2d(1))

        self.roi_pooling = RoIAlignAvg(self.pooling_size, self.pooling_size,
                                       1.0)
        self.anchor_predictor = AnchorPredictor(self.anchor_predictor_config)
        # self.fc_obj = self.__make_fc_layer(
        # [256, 'D', 256, 'D', 2], in_channels=2)
        # self.fc_reg = self.__make_fc_layer(
        # [256, 'D', 256, 'D', 6], in_channels=2)

        # loss
        self.rpn_bbox_loss = nn.SmoothL1Loss(reduce=False)
        if self.use_focal_loss:
            # self.rpn_cls_loss = FocalLoss(
            # self.n_classes, alpha=0.25, gamma=2, auto_alpha=False)
            self.rpn_cls_loss = FocalLoss(self.n_classes)
        else:
            self.rpn_cls_loss = nn.CrossEntropyLoss(reduce=False)

    def __make_fc_layer(self, cfg, in_channels):
        layers = []
        first_layer = True
        for v in cfg:
            if v == 'D':
                layers += [nn.Dropout2d()]
            else:
                if first_layer:
                    layers += [
                        nn.Conv2d(
                            in_channels,
                            v,
                            kernel_size=(self.pooling_size, self.pooling_size))
                    ]
                    in_channels = v
                    first_layer = False
                else:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=1)]
                    in_channels = v

        return nn.Sequential(*layers)

    def init_weights(self):
        # Filler.normal_init(self.bev_feature_extractor, 0, 0.001, self.truncated)
        # Filler.normal_init(self.img_feature_extractor, 0, 0.001, self.truncated)
        # self.bev_feature_extractor.init_weights()
        # self.img_feature_extractor.init_weights()
        # self.anchor_predictor.init_weights()
        pass

    def loss(self, prediction_dict, feed_dict):

        # import ipdb
        # ipdb.set_trace()

        loss_dict = {}
        label_anchors = feed_dict['label_anchors']
        anchors = feed_dict['anchors'].unsqueeze(0)

        bev_anchors_gt_norm = feed_dict['bev_anchors_gt_norm']
        bev_anchors_norm = feed_dict['bev_anchors_norm']
        # bev_shape = feed_dict['bev_feat_maps'].shape[-2:]
        bev_anchors = self.anchors_norm2_anchors(bev_anchors_norm,
                                                 self.bev_shape).unsqueeze(0)
        bev_anchors_gt = self.anchors_norm2_anchors(bev_anchors_gt_norm,
                                                    self.bev_shape)

        #################################
        # target assigner
        ################################
        rcnn_cls_targets, rcnn_reg_targets, \
            rcnn_cls_weights, rcnn_reg_weights = \
                self.target_assigner.assign(
                    bev_anchors,
                    bev_anchors_gt,
                    anchors,
                    label_anchors,
                    gt_labels=None)

        ################################
        # subsample
        ################################
        pos_indicator = rcnn_reg_weights > 0
        indicator = rcnn_cls_weights > 0

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
        rcnn_reg_loss = self.rpn_bbox_loss(rcnn_bbox_preds, rcnn_reg_targets)
        rcnn_reg_loss = rcnn_reg_loss * rcnn_reg_weights.unsqueeze(-1)
        rcnn_reg_loss = rcnn_reg_loss.view(rcnn_reg_loss.shape[0], -1).sum(
            dim=1) / num_reg_coeff.float()

        loss_dict['rpn_cls_loss'] = rcnn_cls_loss
        loss_dict['rpn_bbox_loss'] = rcnn_reg_loss

        return loss_dict
