# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model import Model
# from tmp.core.models.rpn_model import RPNModel
from models.detectors.old_rpn_model import RPNModel
from tmp.core.models.focal_loss import FocalLoss
from tmp.core.models.orientation_loss import OrientationLoss
from lib.model.roi_layers import ROIAlign
#  from model.psroi_pooling.modules.psroi_pool import PSRoIPool

from core.filler import Filler
from tmp.core.mono_3d_target_assigner import TargetAssigner
# from tmp.core.samplers.hard_negative_sampler import HardNegativeSampler
# from tmp.core.samplers.balanced_sampler import BalancedSampler
# from tmp.core.models.feature_extractors.resnet import ResNetFeatureExtractor
# from tmp.core.samplers.detection_sampler import DetectionSampler

import functools
from utils.registry import DETECTORS
from core import constants
import bbox_coders
import samplers
from models import feature_extractors


@DETECTORS.register('mono_3d_better')
class Mono3DFinalPlusFasterRCNN(Model):
    def forward(self, feed_dict):
        self.target_assigner.bbox_coder_3d.mean_dims = feed_dict[constants.
                                                                 KEY_MEAN_DIMS]
        prediction_dict = {}

        # base model
        base_feat = self.feature_extractor.first_stage_feature(
            feed_dict[constants.KEY_IMAGE])
        feed_dict.update({'base_feat': base_feat})

        # rpn model
        prediction_dict.update(self.rpn_model.forward(feed_dict))

        if self.training:
            self.pre_subsample(prediction_dict, feed_dict)
        rois_batch = prediction_dict['rois_batch']

        # note here base_feat (N,C,H,W),rois_batch (N,num_proposals,5)
        pooled_feat = self.rcnn_pooling(base_feat, rois_batch.view(-1, 5))

        # shape(N,C,1,1)
        second_pooled_feat = self.feature_extractor.second_stage_feature(
            pooled_feat)

        second_pooled_feat = second_pooled_feat.mean(3).mean(2)

        rcnn_cls_scores = self.rcnn_cls_preds(second_pooled_feat)
        rcnn_bbox_preds = self.rcnn_bbox_preds(second_pooled_feat)
        rcnn_3d = self.rcnn_3d_pred(second_pooled_feat)

        rcnn_cls_probs = F.softmax(rcnn_cls_scores, dim=1)

        prediction_dict['rcnn_cls_probs'] = rcnn_cls_probs
        prediction_dict['rcnn_bbox_preds'] = rcnn_bbox_preds
        prediction_dict['rcnn_cls_scores'] = rcnn_cls_scores

        # used for track
        proposals_order = prediction_dict['proposals_order']
        prediction_dict['second_rpn_anchors'] = prediction_dict['anchors'][
            proposals_order]

        ###################################
        # 3d training
        ###################################

        prediction_dict['rcnn_3d'] = rcnn_3d

        # if not self.training:
        # _, pred_labels = rcnn_cls_probs.max(dim=-1)
        # rcnn_3d = self.target_assigner.bbox_coder_3d.decode_batch_bbox(
        # rcnn_3d, rois_batch[0, :, 1:])

        # prediction_dict['rcnn_3d'] = rcnn_3d
        # prediction_dict[constants.KEY_CLASSES] = rcnn_cls_probs

        batch_size = feed_dict[constants.KEY_IMAGE].shape[0]
        # coder = bbox_coders.build({'type': constants.KEY_BOXES_2D})
        coder = self.target_assigner.bbox_coder
        proposals = coder.decode_batch(
            rcnn_bbox_preds.view(batch_size, -1, 4),
            rois_batch[0, :, 1:]).detach()

        # used for track
        # proposals_order = prediction_dict['proposals_order']
        # prediction_dict['second_rpn_anchors'] = prediction_dict['anchors'][
        # proposals_order]

        ###################################
        # 3d training
        ###################################

        # prediction_dict['rcnn_3d'] = rcnn_3d

        if not self.training:

            new_rcnn_3d = self.target_assigner.bbox_coder_3d.decode_batch_bbox(
                rcnn_3d, rois_batch[0])

            # prediction_dict['rcnn_3d'] = rcnn_3d
            prediction_dict[constants.KEY_CLASSES] = rcnn_cls_probs.view(
                batch_size, -1, self.n_classes)

            image_info = feed_dict[constants.KEY_IMAGE_INFO]
            proposals[:, :, ::2] = proposals[:, :, ::
                                             2] / image_info[:, 3].unsqueeze(
                                                 -1).unsqueeze(-1)
            proposals[:, :, 1::2] = proposals[:, :, 1::
                                              2] / image_info[:, 2].unsqueeze(
                                                  -1).unsqueeze(-1)
            prediction_dict[constants.KEY_BOXES_2D] = proposals

            #  import ipdb
            #  ipdb.set_trace()
            dims = self.squeeze_bbox_preds(
                new_rcnn_3d[:, :-3].contiguous(),
                rcnn_cls_probs.argmax(dim=-1).view(-1),
                out_c=3).view(batch_size, -1, 3)
            #  import ipdb
            #  ipdb.set_trace()

            #  import ipdb
            #  ipdb.set_trace()
            #  coder = bbox_coders.build({'type': constants.KEY_DIMS})
            #  dims = coder.decode_batch(rcnn_3d[:, :3].view(batch_size, -1, 3),
            #  feed_dict[constants.KEY_MEAN_DIMS],
            #  rcnn_cls_probs).detach()
            rcnn_orient_preds = rcnn_3d[:, 3:].view(batch_size, -1, 5)
            coder = bbox_coders.build({'type': constants.KEY_ORIENTS_V2})
            orients = coder.decode_batch(
                rcnn_orient_preds, proposals,
                feed_dict[constants.KEY_STEREO_CALIB_P2_ORIG]).detach()
            prediction_dict[constants.KEY_DIMS] = dims
            prediction_dict[constants.KEY_ORIENTS_V2] = orients
            prediction_dict['rcnn_3d'] = torch.cat(
                [dims, new_rcnn_3d[:, -3:].view(batch_size, -1, 3)], dim=-1)

        return prediction_dict

    def pre_forward(self):
        pass

    def init_weights(self):
        # submodule init weights
        self.feature_extractor.init_weights()
        self.rpn_model.init_weights()

        Filler.normal_init(self.rcnn_cls_preds, 0, 0.01, self.truncated)
        Filler.normal_init(self.rcnn_bbox_preds, 0, 0.001, self.truncated)

    def init_modules(self):
        # self.feature_extractor = ResNetFeatureExtractor(
        # self.feature_extractor_config)
        self.feature_extractor = feature_extractors.build(
            self.feature_extractor_config)
        self.rpn_model = RPNModel(self.rpn_config)
        if self.pooling_mode == 'align':
            self.rcnn_pooling = ROIAlign((self.pooling_size,
                                          self.pooling_size), 1.0 / 16.0, 2)
        elif self.pooling_mode == 'ps':
            self.rcnn_pooling = PSRoIPool(7, 7, 1.0 / 16, 7, self.n_classes)
        elif self.pooling_mode == 'psalign':
            raise NotImplementedError('have not implemented yet!')
        elif self.pooling_mode == 'deformable_psalign':
            raise NotImplementedError('have not implemented yet!')
        # self.rcnn_cls_pred = nn.Conv2d(2048, self.n_classes, 3, 1, 1)
        self.rcnn_cls_preds = nn.Linear(self.in_channels, self.n_classes)
        if self.class_agnostic:
            self.rcnn_bbox_preds = nn.Linear(self.in_channels, 4)
        else:
            self.rcnn_bbox_preds = nn.Linear(self.in_channels,
                                             4 * self.n_classes)

        # loss module
        if self.use_focal_loss:
            self.rcnn_cls_loss = FocalLoss(self.n_classes)
        else:
            self.rcnn_cls_loss = functools.partial(
                F.cross_entropy, reduce=False)

        self.rcnn_bbox_loss = nn.modules.SmoothL1Loss(reduce=False)

        # self.rcnn_3d_pred = nn.Linear(c, 3 + 4 + 11 + 2 + 1)
        if self.class_agnostic_3d:
            self.rcnn_3d_pred = nn.Linear(self.in_channels, 3 + 5)
        else:
            self.rcnn_3d_pred = nn.Linear(self.in_channels,
                                          3 * self.n_classes + 5)

        self.rcnn_3d_loss = OrientationLoss(split_loss=True)

    def init_param(self, model_config):
        self.in_channels = model_config.get('ndin', 2048)
        classes = model_config['classes']
        self.classes = classes
        self.n_classes = len(classes) + 1
        self.class_agnostic = model_config['class_agnostic']
        self.pooling_size = model_config['pooling_size']
        self.pooling_mode = model_config['pooling_mode']
        self.class_agnostic_3d = model_config['class_agnostic_3d']
        self.crop_resize_with_max_pool = model_config[
            'crop_resize_with_max_pool']
        self.truncated = model_config['truncated']

        self.use_focal_loss = model_config['use_focal_loss']
        self.subsample_twice = model_config['subsample_twice']
        self.rcnn_batch_size = model_config['rcnn_batch_size']

        # some submodule config
        # self.feature_extractor_config = model_config['feature_extractor_config']
        self.feature_extractor_config = {
            "type": "resnet",
            "pretrained_path": "./data/pretrained_model",
            "net_arch": "res18_pruned",
            "separate_feat": False,
            "use_cascade": True,
            "class_agnostic": True,
            "classes":
            ["Car", "Van", "Truck", "Pedestrian", "Cyclist", "Tram"],
            "img_channels": 3,
            "pretrained_model": "",
            "pretrained": True
        }
        self.rpn_config = model_config['rpn_config']

        # sampler
        # self.sampler = BalancedSampler(model_config['sampler_config'])
        self.sampler = samplers.build({
            "num_samples": 512,
            "type": "balanced",
            "fg_fraction": 0.5
        })

        # self.reduce = model_config.get('reduce')
        self.reduce = True
        # more accurate bbox for 3d prediction
        # if self.train_3d:
        # fg_thresh = 0.6
        # else:
        # fg_thresh = 0.5
        # model_config['target_assigner_config']['fg_thresh'] = fg_thresh

        # assigner
        self.target_assigner = TargetAssigner(
            model_config['target_assigner_config'])

    def pre_subsample(self, prediction_dict, feed_dict):
        rois_batch = prediction_dict['rois_batch']
        gt_boxes = feed_dict[constants.KEY_LABEL_BOXES_2D]
        # gt_boxes_proj = feed_dict['gt_boxes_proj']
        gt_labels = feed_dict[constants.KEY_LABEL_CLASSES]

        # shape(N,7)
        gt_boxes_3d = feed_dict[constants.KEY_LABEL_BOXES_3D]

        # orient
        cls_orient = torch.unsqueeze(feed_dict['cls_orient'], dim=-1).float()
        reg_orient = feed_dict['reg_orient']
        orient = torch.cat([cls_orient, reg_orient], dim=-1)

        gt_boxes_3d = torch.cat([gt_boxes_3d[:, :, :3], orient], dim=-1)

        ##########################
        # assigner
        ##########################
        rcnn_cls_targets, rcnn_reg_targets,\
            rcnn_cls_weights, rcnn_reg_weights,\
            rcnn_reg_targets_3d, rcnn_reg_weights_3d = self.target_assigner.assign(
            rois_batch[:, :, 1:], gt_boxes, gt_boxes_3d, gt_labels)

        ##########################
        # subsampler
        ##########################
        cls_criterion = None
        pos_indicator = rcnn_reg_weights > 0
        indicator = rcnn_cls_weights > 0

        # subsample from all
        # shape (N,M)
        batch_sampled_mask = self.sampler.subsample_batch(
            pos_indicator, indicator=indicator, criterion=cls_criterion)
        rcnn_cls_weights = rcnn_cls_weights[batch_sampled_mask]
        rcnn_reg_weights = rcnn_reg_weights[batch_sampled_mask]
        rcnn_reg_weights_3d = rcnn_reg_weights_3d[batch_sampled_mask]
        num_cls_coeff = (rcnn_cls_weights > 0).sum(dim=-1)
        num_reg_coeff = (rcnn_reg_weights > 0).sum(dim=-1)
        # check
        assert num_cls_coeff, 'bug happens'
        # assert num_reg_coeff, 'bug happens'
        if num_reg_coeff == 0:
            num_reg_coeff = torch.ones_like(num_reg_coeff)

        prediction_dict[
            'rcnn_cls_weights'] = rcnn_cls_weights / num_cls_coeff.float()
        prediction_dict[
            'rcnn_reg_weights'] = rcnn_reg_weights / num_reg_coeff.float()
        prediction_dict[
            'rcnn_reg_weights_3d'] = rcnn_reg_weights_3d / num_reg_coeff.float()
        prediction_dict['rcnn_cls_targets'] = rcnn_cls_targets[
            batch_sampled_mask]
        prediction_dict['rcnn_reg_targets'] = rcnn_reg_targets[
            batch_sampled_mask]
        prediction_dict['rcnn_reg_targets_3d'] = rcnn_reg_targets_3d[
            batch_sampled_mask]

        # update rois_batch
        prediction_dict['rois_batch'] = rois_batch[batch_sampled_mask].view(
            rois_batch.shape[0], -1, 5)

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

    def loss(self, prediction_dict, feed_dict):
        """
        assign proposals label and subsample from them
        Then calculate loss
        """
        #  import ipdb
        #  ipdb.set_trace()

        loss_dict = {}

        # submodule loss
        loss_dict.update(self.rpn_model.loss(prediction_dict, feed_dict))
        # targets and weights
        rcnn_cls_weights = prediction_dict['rcnn_cls_weights']
        rcnn_reg_weights = prediction_dict['rcnn_reg_weights']

        rcnn_cls_targets = prediction_dict['rcnn_cls_targets']
        rcnn_reg_targets = prediction_dict['rcnn_reg_targets']

        # classification loss
        rcnn_cls_scores = prediction_dict['rcnn_cls_scores']

        rcnn_cls_loss = self.rcnn_cls_loss(rcnn_cls_scores, rcnn_cls_targets)
        rcnn_cls_loss *= rcnn_cls_weights
        rcnn_cls_loss = rcnn_cls_loss.sum(dim=-1)

        # bounding box regression L1 loss
        rcnn_bbox_preds = prediction_dict['rcnn_bbox_preds']
        #
        if not self.class_agnostic:
            rcnn_bbox_preds = self.squeeze_bbox_preds(rcnn_bbox_preds,
                                                      rcnn_cls_targets)
        rcnn_bbox_loss = self.rcnn_bbox_loss(rcnn_bbox_preds,
                                             rcnn_reg_targets).sum(dim=-1)
        rcnn_bbox_loss *= rcnn_reg_weights
        rcnn_bbox_loss = rcnn_bbox_loss.sum(dim=-1)

        loss_dict['rcnn_cls_loss'] = rcnn_cls_loss
        loss_dict['rcnn_bbox_loss'] = rcnn_bbox_loss

        ######################################
        # 3d loss
        ######################################

        rcnn_reg_weights_3d = prediction_dict['rcnn_reg_weights_3d']
        rcnn_reg_targets_3d = prediction_dict['rcnn_reg_targets_3d']
        rcnn_3d = prediction_dict['rcnn_3d']

        if not self.class_agnostic_3d:
            dims_pred = rcnn_3d[:, :3 * self.n_classes]
            dims_pred = self.squeeze_bbox_preds(dims_pred, rcnn_cls_targets, 3)
            orient_pred = rcnn_3d[:, 3 * self.n_classes:]
        else:
            dims_pred = rcnn_3d[:, :3]
            orient_pred = rcnn_3d[:, 3:]
        # dims
        rcnn_3d_loss_dims = self.rcnn_bbox_loss(
            dims_pred, rcnn_reg_targets_3d[:, :3]).sum(dim=-1)

        # angles
        res = self.rcnn_3d_loss(orient_pred, rcnn_reg_targets_3d[:, 3:6])
        for res_loss_key in res:
            tmp = res[res_loss_key] * rcnn_reg_weights_3d
            res[res_loss_key] = tmp.sum(dim=-1)
        loss_dict.update(res)

        rcnn_3d_loss = rcnn_3d_loss_dims * rcnn_reg_weights_3d
        rcnn_3d_loss = rcnn_3d_loss.sum(dim=-1)

        loss_dict['rcnn_3d_loss'] = rcnn_3d_loss

        # stats of orients
        cls_orient_preds = rcnn_3d[:, 3:5]
        cls_orient = rcnn_reg_targets_3d[:, 3]
        _, cls_orient_preds_argmax = torch.max(cls_orient_preds, dim=-1)
        orient_tp_mask = cls_orient.type_as(
            cls_orient_preds_argmax) == cls_orient_preds_argmax
        mask = (rcnn_reg_weights_3d > 0) & (rcnn_reg_targets_3d[:, 3] > -1)
        orient_tp_mask = orient_tp_mask[mask]
        orient_tp_num = orient_tp_mask.int().sum().item()
        orient_all_num = orient_tp_mask.numel()

        # gt_boxes_proj = feed_dict['gt_boxes_proj']

        # self.target_assigner.stat.update({
        # # 'angle_num_tp': torch.tensor(0),
        # # 'angle_num_all': 1,

        # # stats of orient
        # 'orient_tp_num': orient_tp_num,
        # # 'orient_tp_num2': orient_tp_num2,
        # #  'orient_tp_num3': orient_tp_num3,
        # # 'orient_all_num3': orient_all_num3,
        # # 'orient_pr': orient_pr,
        # 'orient_all_num': orient_all_num,
        # #  'orient_all_num3': orient_all_num3,
        # # 'orient_tp_num4': orient_tp_num4,
        # # 'orient_all_num4': orient_all_num4,
        # #  'cls_orient_2s_all_num': depth_ind_all_num,
        # #  'cls_orient_2s_tp_num': depth_ind_tp_num
        # })
        prediction_dict[constants.KEY_STATS] = [{
            constants.KEY_STATS_ORIENTS_PRECISION:
            torch.tensor([orient_tp_num, orient_all_num]).to(
                'cuda').float().unsqueeze(0)
        }]

        return loss_dict
