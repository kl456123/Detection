# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
import torch.nn.functional as F

from core.model import Model
from models.detectors.rpn_model import RPNModel as _RPNModel
from core.filler import Filler
from models.losses.focal_loss import FocalLoss

from utils import box_ops
from lib.model.roi_layers import nms
from utils.registry import DETECTORS
from target_generators.target_generator import TargetGenerator
# import samplers
from models.losses import common_loss
from core import constants
import bbox_coders

from core.utils import tensor_utils
from utils import geometry_utils


@DETECTORS.register('fpn_rpn_grnet')
class RPNModel(_RPNModel):
    def generate_proposal(self, rpn_cls_probs, anchors, rpn_bbox_preds,
                          im_info):
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
        # assert len(
        # rpn_bbox_preds) == 1, 'just one feature maps is supported now'
        # rpn_bbox_preds = rpn_bbox_preds[0]
        # do not backward
        rpn_cls_probs = rpn_cls_probs.detach()
        rpn_bbox_preds = rpn_bbox_preds.detach()

        batch_size = rpn_bbox_preds.shape[0]

        coders = bbox_coders.build(
            self.target_generators.target_generator_config['coder_config'])
        proposals = coders.decode_batch(rpn_bbox_preds, anchors)

        # filer and clip
        proposals = box_ops.clip_boxes(proposals, im_info)

        # fg prob
        fg_probs = rpn_cls_probs[:, :, 1]

        # sort fg
        _, fg_probs_order = torch.sort(fg_probs, dim=1, descending=True)

        # fg_probs_batch = torch.zeros(batch_size,
        # self.post_nms_topN).type_as(rpn_cls_probs)
        proposals_batch = torch.zeros(batch_size, self.post_nms_topN,
                                      4).type_as(rpn_bbox_preds)
        proposals_order = torch.zeros(
            batch_size, self.post_nms_topN).fill_(-1).type_as(fg_probs_order)

        for i in range(batch_size):
            proposals_single = proposals[i]
            fg_probs_single = fg_probs[i]
            fg_order_single = fg_probs_order[i]
            # pre nms
            if self.pre_nms_topN > 0:
                fg_order_single = fg_order_single[:self.pre_nms_topN]
            proposals_single = proposals_single[fg_order_single]
            fg_probs_single = fg_probs_single[fg_order_single]

            # nms
            keep_idx_i = nms(proposals_single, fg_probs_single,
                             self.nms_thresh)
            keep_idx_i = keep_idx_i.long().view(-1)

            # post nms
            if self.post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:self.post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            fg_probs_single = fg_probs_single[keep_idx_i]
            fg_order_single = fg_order_single[keep_idx_i]

            # padding 0 at the end.
            num_proposal = keep_idx_i.numel()
            proposals_batch[i, :num_proposal, :] = proposals_single
            # fg_probs_batch[i, :num_proposal] = fg_probs_single
            proposals_order[i, :num_proposal] = fg_order_single
        return proposals_batch, proposals_order

    def init_modules(self):
        super().init_modules()
        self.rpn_corners_preds = nn.Conv2d(512, 7 * self.num_anchors, 1, 1, 0)
        self.l1_loss = nn.L1Loss(reduction='none')
        self.l2_loss = nn.MSELoss(reduction='none')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, bottom_blobs):
        rpn_feat_maps = bottom_blobs['rpn_feat_maps']
        batch_size = rpn_feat_maps[0].shape[0]
        im_info = bottom_blobs[constants.KEY_IMAGE_INFO]

        rpn_cls_scores = []
        # rpn_cls_probs = []
        rpn_bbox_preds = []
        rpn_corners_preds = []

        for rpn_feat_map in rpn_feat_maps:
            # rpn conv
            rpn_conv = F.relu(self.rpn_conv(rpn_feat_map), inplace=True)

            # rpn cls score
            # shape(N,2*num_anchors,H,W)
            rpn_cls_score = self.rpn_cls_score(rpn_conv)
            rpn_cls_score = rpn_cls_score.permute(0, 2, 3, 1).contiguous()
            rpn_cls_score = rpn_cls_score.view(batch_size, -1, 2)

            # rpn cls prob shape(N,2*num_anchors,H,W)
            # rpn_cls_score_reshape = rpn_cls_score.view(batch_size, 2, -1)
            # rpn_cls_prob = F.softmax(rpn_cls_score_reshape, dim=1)
            # rpn_cls_prob = rpn_cls_prob.view_as(rpn_cls_score)

            # rpn_cls_prob = rpn_cls_prob.view(batch_size, 2, -1,
            # rpn_cls_prob.shape[2],
            # rpn_cls_prob.shape[3])
            # rpn_cls_prob = rpn_cls_prob.permute(
            # 0, 3, 4, 2, 1).contiguous().view(batch_size, -1, 2)

            # rpn_cls_score = rpn_cls_score.view(batch_size, 2, -1,
            # rpn_cls_score.shape[2],
            # rpn_cls_score.shape[3])
            # rpn_cls_score = rpn_cls_score.permute(
            # 0, 3, 4, 2, 1).contiguous().view(batch_size, -1, 2)
            # bbox
            rpn_bbox_pred = self.rpn_bbox_pred(rpn_conv)
            rpn_bbox_pred = rpn_bbox_pred.permute(0, 2, 3, 1).contiguous()
            rpn_bbox_pred = rpn_bbox_pred.view(batch_size, -1, 4)

            # corners
            rpn_corners_pred = self.rpn_corners_preds(rpn_conv)
            rpn_corners_pred = rpn_corners_pred.permute(0, 2, 3,
                                                        1).contiguous()
            rpn_corners_pred = rpn_corners_pred.view(batch_size, -1, 7)

            rpn_cls_scores.append(rpn_cls_score)
            rpn_bbox_preds.append(rpn_bbox_pred)
            rpn_corners_preds.append(rpn_corners_pred)

        rpn_cls_scores = torch.cat(rpn_cls_scores, dim=1)
        # rpn_cls_probs = torch.cat(rpn_cls_probs, dim=1)
        rpn_bbox_preds = torch.cat(rpn_bbox_preds, dim=1)
        rpn_corners_preds = torch.cat(rpn_corners_preds, dim=1)
        rpn_cls_probs = F.softmax(rpn_cls_scores, dim=-1)

        # generate pyramid anchors
        feature_map_list = [
            base_feat.shape[-2:] for base_feat in rpn_feat_maps
        ]
        anchors = self.anchor_generator.generate_pyramid(
            feature_map_list, im_info[0][:-1])
        anchors = anchors.unsqueeze(0).repeat(batch_size, 1, 1)

        ###############################
        # Proposal
        ###############################
        # note that proposals_order is used for track transform of propsoals
        proposals_batch, proposals_order = self.generate_proposal(
            rpn_cls_probs, anchors, rpn_bbox_preds, im_info)
        # import ipdb
        # ipdb.set_trace()
        rpn_corners_preds_detach = tensor_utils.multidim_index(
            rpn_corners_preds, proposals_order).detach()

        # if self.training:
        # label_boxes_2d = bottom_blobs[constants.KEY_LABEL_BOXES_2D]
        # proposals_batch = self.append_gt(proposals_batch, label_boxes_2d)

        # postprocess

        predict_dict = {
            'proposals': proposals_batch,
            'rpn_cls_scores': rpn_cls_scores,
            'anchors': anchors,

            # used for loss
            'rpn_bbox_preds': rpn_bbox_preds,
            'rpn_cls_probs': rpn_cls_probs,
            constants.KEY_CORNERS_3D_GRNET: rpn_corners_preds_detach,
            'corners_3d':rpn_corners_preds
        }

        return predict_dict

    def calc_local_corners(self, dims, ry):
        # import ipdb
        # ipdb.set_trace()
        h = dims[:, 0]
        w = dims[:, 1]
        l = dims[:, 2]
        zeros = torch.zeros_like(l).type_as(l)
        # rotation_matrix = geometry_utils.torch_ry_to_rotation_matrix(ry)

        zeros = torch.zeros_like(ry[:, 0])
        ones = torch.ones_like(ry[:, 0])
        cos = torch.cos(ry[:, 0])
        sin = torch.sin(ry[:, 0])
        # norm = torch.norm(ry, dim=-1)
        cos = cos
        sin = sin

        rotation_matrix = torch.stack(
            [cos, zeros, sin, zeros, ones, zeros, -sin, zeros, cos],
            dim=-1).reshape(-1, 3, 3)

        x_corners = torch.stack(
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            dim=0)
        y_corners = torch.stack(
            [zeros, zeros, zeros, zeros, -h, -h, -h, -h], dim=0)
        z_corners = torch.stack(
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            dim=0)

        # shape(N, 3, 8)
        box_points_coords = torch.stack(
            (x_corners, y_corners, z_corners), dim=0)
        # rotate and translate
        # shape(N, 3, 8)
        corners_3d = torch.bmm(rotation_matrix,
                               box_points_coords.permute(2, 0, 1))

        return corners_3d.permute(0, 2, 1)

    def decode_center_depth(self, dims_preds, final_boxes_2d_xywh, p2):
        f = p2[:, 0, 0]
        h_2d = final_boxes_2d_xywh[:, :, -1] + 1e-6
        h_3d = dims_preds[:, :, 0]
        depth_preds = f.unsqueeze(-1) * h_3d / h_2d
        return depth_preds.unsqueeze(-1)

    def loss(self, prediction_dict, feed_dict):
        # import ipdb
        # ipdb.set_trace()
        loss_dict = {}
        anchors = prediction_dict['anchors']
        anchors_dict = {}
        anchors_dict[constants.KEY_PRIMARY] = anchors
        anchors_dict[constants.KEY_BOXES_2D] = prediction_dict[
            'rpn_bbox_preds']
        anchors_dict[constants.KEY_CLASSES] = prediction_dict['rpn_cls_scores']
        anchors_dict[constants.KEY_CORNERS_3D_GRNET] = prediction_dict['corners_3d']

        gt_dict = {}
        gt_dict[constants.KEY_PRIMARY] = feed_dict[
            constants.KEY_LABEL_BOXES_2D]
        gt_dict[constants.KEY_CLASSES] = None
        gt_dict[constants.KEY_BOXES_2D] = None
        gt_dict[constants.KEY_CORNERS_3D_GRNET] = None

        auxiliary_dict = {}
        auxiliary_dict[constants.KEY_BOXES_2D] = feed_dict[
            constants.KEY_LABEL_BOXES_2D]
        gt_labels = feed_dict[constants.KEY_LABEL_CLASSES]
        auxiliary_dict[constants.KEY_CLASSES] = torch.ones_like(gt_labels)
        auxiliary_dict[constants.KEY_NUM_INSTANCES] = feed_dict[
            constants.KEY_NUM_INSTANCES]
        auxiliary_dict[constants.KEY_PROPOSALS] = anchors
        auxiliary_dict[constants.KEY_BOXES_3D] = feed_dict[
            constants.KEY_LABEL_BOXES_3D]
        auxiliary_dict[constants.KEY_STEREO_CALIB_P2] = feed_dict[
            constants.KEY_STEREO_CALIB_P2]

        # import ipdb
        # ipdb.set_trace()
        subsample = not self.use_focal_loss
        _, targets, _ = self.target_generators.generate_targets(
            anchors_dict, gt_dict, auxiliary_dict, subsample=subsample)

        cls_target = targets[constants.KEY_CLASSES]
        reg_target = targets[constants.KEY_BOXES_2D]

        # loss

        if self.use_focal_loss:
            # when using focal loss, dont normalize it by all samples
            cls_targets = cls_target['target']
            pos = cls_targets > 0  # [N,#anchors]
            num_pos = pos.long().sum().clamp(min=1).float()
            rpn_cls_loss = common_loss.calc_loss(
                self.rpn_cls_loss, cls_target, normalize=False) / num_pos
        else:
            rpn_cls_loss = common_loss.calc_loss(self.rpn_cls_loss, cls_target)
        rpn_reg_loss = common_loss.calc_loss(self.rpn_bbox_loss, reg_target)
        loss_dict.update({
            'rpn_cls_loss': rpn_cls_loss,
            'rpn_reg_loss': rpn_reg_loss
        })

        # return loss_dict
        # super().loss(prediction_dict, feed_dict)


        # proposals = prediction_dict[constants.KEY_PROPOSALS]
        proposals = anchors_dict[constants.KEY_PRIMARY]
        p2 = feed_dict[constants.KEY_STEREO_CALIB_P2]
        image_info = feed_dict[constants.KEY_IMAGE_INFO]
        mean_dims = torch.tensor([1.8, 1.8, 3.7]).type_as(proposals)
        corners_2d_loss = 0
        center_depth_loss = 0
        location_loss = 0

        corners_target = targets[constants.KEY_CORNERS_3D_GRNET]
        # rcnn_corners_loss = rcnn_corners_loss + common_loss.calc_loss(
        # self.rcnn_corners_loss, orient_target, True)
        preds = corners_target['pred']
        targets = corners_target['target']
        weights = corners_target['weight']
        weights = weights.unsqueeze(-1)

        local_corners_gt = targets[:, :, :24]
        location_gt = targets[:, :, 24:27]
        dims_gt = targets[:, :, 27:]
        N, M = local_corners_gt.shape[:2]

        global_corners_gt = (local_corners_gt.view(N, M, 8, 3) +
                             location_gt.view(N, M, 1, 3)).view(N, M, -1)
        center_depth_gt = location_gt[:, :, 2:]

        dims_preds = torch.exp(preds[:, :, :3]) * mean_dims
        # import ipdb
        # ipdb.set_trace()
        dims_loss = self.l1_loss(dims_preds, dims_gt) * weights
        ry_preds = preds[:, :, 3:4]
        # ray_angle = -torch.atan2(location_gt[:, :, 2], location_gt[:, :, 0])
        # ry_preds = ry_preds + ray_angle.unsqueeze(-1)
        local_corners_preds = []
        # calc local corners preds
        for batch_ind in range(N):
            local_corners_preds.append(
                self.calc_local_corners(dims_preds[batch_ind].detach(),
                                        ry_preds[batch_ind]))
        local_corners_preds = torch.stack(local_corners_preds, dim=0)

        center_2d_deltas_preds = preds[:, :, 4:6]
        center_depth_preds = preds[:, :, 6:]
        # import ipdb
        # ipdb.set_trace()
        # decode center_2d
        proposals_xywh = geometry_utils.torch_xyxy_to_xywh(proposals)
        center_depth_init = self.decode_center_depth(dims_preds,
                                                     proposals_xywh, p2)
        center_depth_preds = center_depth_init * center_depth_preds
        center_2d_preds = (center_2d_deltas_preds * proposals_xywh[:, :, 2:] +
                           proposals_xywh[:, :, :2])
        # center_depth_preds_detach = center_depth_preds.detach()

        # import ipdb
        # ipdb.set_trace()
        # use gt depth to cal loss to make sure the gradient smooth
        location_preds = []
        for batch_ind in range(N):
            location_preds.append(
                geometry_utils.torch_points_2d_to_points_3d(
                    center_2d_preds[batch_ind], center_depth_gt[batch_ind],
                    p2[batch_ind]))
        location_preds = torch.stack(location_preds, dim=0)
        global_corners_preds = (location_preds.view(N, M, 1, 3) +
                                local_corners_preds.view(N, M, 8, 3)).view(
                                    N, M, -1)

        # import ipdb
        # ipdb.set_trace()
        # corners depth loss and center depth loss
        corners_depth_preds = local_corners_preds.view(N, M, 8, 3)[..., -1]
        corners_depth_gt = local_corners_gt.view(N, M, 8, 3)[..., -1]

        center_depth_loss = self.l1_loss(center_depth_preds,
                                         center_depth_gt) * weights

        # location loss
        location_loss = self.l1_loss(location_preds, location_gt) * weights

        # global corners loss
        global_corners_loss = self.l1_loss(global_corners_preds,
                                           global_corners_gt) * weights

        # proj 2d loss
        corners_2d_preds = []
        corners_2d_gt = []
        for batch_ind in range(N):
            corners_2d_preds.append(
                geometry_utils.torch_points_3d_to_points_2d(
                    global_corners_preds[batch_ind].view(-1, 3),
                    p2[batch_ind]))
            corners_2d_gt.append(
                geometry_utils.torch_points_3d_to_points_2d(
                    global_corners_gt[batch_ind].view(-1, 3), p2[batch_ind]))

        corners_2d_preds = torch.stack(corners_2d_preds, dim=0).view(N, M, -1)
        corners_2d_gt = torch.stack(corners_2d_gt, dim=0).view(N, M, -1)

        # image filter
        # import ipdb
        # ipdb.set_trace()
        zeros = torch.zeros_like(image_info[:, 0])
        image_shape = torch.stack(
            [zeros, zeros, image_info[:, 1], image_info[:, 0]], dim=-1)
        image_shape = image_shape.type_as(corners_2d_gt).view(-1, 4)
        image_filter = geometry_utils.torch_window_filter(
            corners_2d_gt.view(N, -1, 2), image_shape,
            deltas=200).float().view(N, M, -1)

        # import ipdb
        # ipdb.set_trace()
        encoded_corners_2d_gt = corners_2d_gt.view(N, M, 8, 2)
        encoded_corners_2d_preds = corners_2d_preds.view(N, M, 8, 2)
        corners_2d_loss = self.l2_loss(
            encoded_corners_2d_preds.view(N, M, -1),
            encoded_corners_2d_gt.view(N, M, -1)) * weights
        corners_2d_loss = (
            corners_2d_loss.view(N, M, 8, 2) * image_filter.unsqueeze(-1))
        # import ipdb
        # ipdb.set_trace()
        # mask = self.select_corners(global_corners_gt)
        # mask = mask.unsqueeze(-1).expand_as(corners_2d_loss).float()
        corners_2d_loss = corners_2d_loss.view(N, M, -1)
        corners_depth_loss = self.l1_loss(
            corners_depth_preds, corners_depth_gt) * weights * image_filter

        # import ipdb
        # ipdb.set_trace()
        # corners_3d_gt = []
        # for batch_ind in range(N):
        # corners_3d_gt.append(
        # geometry_utils.torch_points_2d_to_points_3d(
        # corners_2d_preds[batch_ind].view(-1, 2),
        # corners_depth_preds[batch_ind].view(-1), p2[batch_ind]))
        # corners_3d_gt = torch.stack(corners_3d_gt, dim=0).view(N, M, -1)

        # dim_target = targets[stage_ind][3]
        # rcnn_dim_loss = rcnn_dim_loss + common_loss.calc_loss(
        # self.rcnn_bbox_loss, dim_target, True)

        global_corners_loss = self.l1_loss(global_corners_preds,
                                           global_corners_gt) * weights
        # local_corners_loss = self.l1_loss(local_corners_preds,
        # local_corners_gt) * weights
        # import ipdb
        # ipdb.set_trace()
        num_pos = (weights > 0).long().sum().clamp(min=1).float()

        loss_dict.update({
            # 'global_corners_loss': global_corners_loss,
            # 'local_corners_loss': local_corners_loss * 10,
            'corners_2d_loss': corners_2d_loss,
            # 'center_depth_loss': center_depth_loss,
            # 'location_loss': location_loss,
            # 'corners_depth_loss': corners_depth_loss * 10,
            # 'rcnn_corners_loss': rcnn_corners_loss,
            # 'rcnn_dim_loss': rcnn_dim_loss
            # 'dims_loss': dims_loss
        })

        return loss_dict
