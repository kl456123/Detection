# -*- coding: utf-8 -*-

from core.model import Model
import torch.nn as nn
import torch
from models.feature_extractors.prnet import PRNetFeatureExtractor
from target_generators.target_generator import TargetGenerator
from core import constants
from models.losses import common_loss
from models.losses.two_step_focal_loss import FocalLoss as TwoStepFocalLoss
from models.losses.focal_loss import FocalLoss
from models.losses.retinanet_loss import FocalLoss as RetinaNetLoss
from models.losses.focal_loss_old import FocalLoss as OldFocalLoss
from models.losses.corners_loss import CornersLoss
import anchor_generators
from utils.registry import DETECTORS
import bbox_coders
import torch.nn.functional as F
from core.utils.analyzer import Analyzer
from models.losses.orientation_loss import OrientationLoss

from utils import geometry_utils


def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]  # [N,D]


def focal_loss_alt(num_classes):
    def focal_loss_alt(x, y):
        '''Focal loss alternative.
            Args:
            x: (tensor) sized [N,D].
            y: (tensor) sized [N,].
            Return:
            (tensor) focal loss.
            '''
        alpha = 0.25

        t = one_hot_embedding(y.data.cpu(), num_classes)
        t = t.cuda()

        xt = x * (2 * t - 1)  # xt = x if t > 0 else -x
        pt = (2 * xt + 1).sigmoid()

        w = alpha * t + (1 - alpha) * (1 - t)
        loss = -w * pt.log() / 2
        return loss.sum(dim=-1)

    return focal_loss_alt


@DETECTORS.register('prnet_mono_3d')
class TwoStageRetinaLayer(Model):
    def init_param(self, model_config):
        # including bg
        self.num_classes = len(model_config['classes']) + 1
        self.in_channels = model_config.get('in_channels', 128)
        self.num_regress = model_config.get('num_regress', 4)
        self.feature_extractor_config = model_config[
            'feature_extractor_config']

        self.target_generators = TargetGenerator(
            model_config['target_generator_config'])
        self.anchor_generator = anchor_generators.build(
            model_config['anchor_generator_config'])

        self.num_anchors = self.anchor_generator.num_anchors
        input_size = torch.tensor(model_config['input_size']).float()
        self.anchors = self.anchor_generator.generate(input_size)

        self.use_focal_loss = model_config['use_focal_loss']

    def init_modules(self):

        in_channels = self.in_channels

        self.loc_feature1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.loc_feature2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.cls_feature1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.cls_feature2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.corners_feature = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # self.orient_feature = nn.Sequential(
        # nn.Conv2d(
        # in_channels, in_channels, kernel_size=1, stride=1),
        # nn.Conv2d(
        # in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        # nn.ReLU(inplace=True),
        # nn.Conv2d(
        # in_channels, in_channels, kernel_size=3, stride=1, padding=1),
        # nn.ReLU(inplace=True), )

        self.os_out = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_anchors * 2,
            kernel_size=1)
        self.cls_out = nn.Conv2d(
            in_channels, self.num_anchors * self.num_classes, kernel_size=1)
        self.box_out1 = nn.Conv2d(
            in_channels,
            out_channels=self.num_anchors * self.num_regress,
            kernel_size=1)
        self.box_out2 = nn.Conv2d(
            in_channels,
            out_channels=self.num_anchors * self.num_regress,
            kernel_size=1)
        self.corners_out = nn.Conv2d(
            in_channels,
            out_channels=self.num_anchors * (2 + 4 + 1),
            kernel_size=1)
        # self.orient_out = nn.Conv2d(
        # in_channels, out_channels=self.num_anchors * 5, kernel_size=1)

        self.feature_extractor = PRNetFeatureExtractor(
            self.feature_extractor_config)

        self.two_step_loss = TwoStepFocalLoss(self.num_classes)
        self.rpn_bbox_loss = nn.SmoothL1Loss(reduction='none')
        if self.use_focal_loss:
            # optimized too slowly
            self.rpn_cls_loss = OldFocalLoss(self.num_classes)
            # fg or bg
            self.rpn_os_loss = OldFocalLoss(2)
        else:
            self.rpn_cls_loss = nn.CrossEntropyLoss(reduction='none')
            self.rpn_os_loss = nn.CrossEntropyLoss(reduction='none')

        self.retina_loss = RetinaNetLoss(self.num_classes)
        self.l1_loss = nn.L1Loss(reduction='none')
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='none')

        # self.rcnn_corners_loss = CornersLoss(
        # use_filter=True, training_depth=False)

        # self.rcnn_orient_preds = nn.Linear(1024, 5)

        # self.rcnn_dim_preds = nn.Linear(1024, 3)

        # self.rcnn_orient_loss = OrientationLoss()

        # self.rcnn_corners_preds = nn.Linear(1024, 41)

    def forward(self, feed_dict):
        features = self.feature_extractor(feed_dict[constants.KEY_IMAGE])
        y_locs1 = []
        y_locs2 = []
        y_os = []
        y_cls = []
        # y_dims = []
        # y_orients = []
        y_corners = []

        for i, x in enumerate(features):
            # location out
            loc_feature = self.loc_feature1(x)
            loc1 = self.box_out1(loc_feature)

            N = loc1.size(0)
            loc1 = loc1.permute(0, 2, 3, 1).contiguous()
            loc1 = loc1.view(N, -1, self.num_regress)
            y_locs1.append(loc1)

            loc_feature = torch.cat([x, loc_feature], dim=1)
            loc_feature = self.loc_feature2(loc_feature)
            loc2 = self.box_out2(loc_feature)

            N = loc2.size(0)
            loc2 = loc2.permute(0, 2, 3, 1).contiguous()
            loc2 = loc2.view(N, -1, self.num_regress)
            loc2 += loc1
            y_locs2.append(loc2)

            # os out
            cls_feature = self.cls_feature1(x)
            os_out = self.os_out(cls_feature)
            os_out = os_out.permute(0, 2, 3, 1).contiguous()
            # _size = os_out.size(1)
            os_out = os_out.view(N, -1, 2)
            y_os.append(os_out)

            cls_feature = torch.cat([x, cls_feature], dim=1)
            cls_feature = self.cls_feature2(cls_feature)
            cls_out = self.cls_out(cls_feature)

            cls_out = cls_out.permute(0, 2, 3, 1).contiguous()
            cls_out = cls_out.view(N, -1, self.num_classes)
            y_cls.append(cls_out)

            # dim out
            corners_feature = self.corners_feature(x)
            corners_out = self.corners_out(corners_feature)
            corners_out = corners_out.permute(0, 2, 3, 1).contiguous()
            corners_out = corners_out.view(N, -1, 7)
            y_corners.append(corners_out)

            # orient out
            # orient_feature = self.orient_feature(x)
            # orient_out = self.orient_out(orient_feature)
            # orient_out = orient_out.permute(0, 2, 3, 1).contiguous()
            # orient_out = orient_out.view(N, -1, 5)
            # y_orients.append(orient_out)

        loc1_preds = torch.cat(y_locs1, dim=1)
        loc2_preds = torch.cat(y_locs2, dim=1)
        os_preds = torch.cat(y_os, dim=1)
        cls_preds = torch.cat(y_cls, dim=1)
        # orient_preds = torch.cat(y_orients, dim=1)
        corners_preds = torch.cat(y_corners, dim=1)

        image_info = feed_dict[constants.KEY_IMAGE_INFO]

        batch_size = loc1_preds.shape[0]

        anchors = self.anchors.cuda()
        anchors = anchors.repeat(batch_size, 1, 1)

        coder = bbox_coders.build(
            self.target_generators.target_generator_config['coder_config'])
        proposals = coder.decode_batch(loc2_preds, anchors).detach()

        cls_probs = F.softmax(cls_preds.detach(), dim=-1)
        os_probs = F.softmax(os_preds.detach(), dim=-1)[:, :, 1:]
        os_probs[os_probs <= 0.4] = 0
        final_probs = cls_probs * os_probs
        # import ipdb
        # ipdb.set_trace()
        #  final_probs = cls_probs

        coder = bbox_coders.build({'type': constants.KEY_CORNERS_3D_GRNET})
        # decoded_dim_preds = coder.decode_batch(
        # dim_preds, feed_dict[constants.KEY_MEAN_DIMS],
        # final_probs).detach()
        # coder = bbox_coders.build({'type': constants.KEY_ORIENTS_V2})
        # # use rpn proposals to decode
        # decoded_orient_preds = coder.decode_batch(
        # orient_preds, proposals,
        # feed_dict[constants.KEY_STEREO_CALIB_P2]).detach()

        prediction_dict = {}
        if self.training:

            # anchors = prediction_dict['anchors']
            anchors_dict = {}
            anchors_dict[constants.KEY_PRIMARY] = anchors
            anchors_dict[constants.KEY_BOXES_2D] = loc1_preds
            anchors_dict[constants.KEY_BOXES_2D_REFINE] = loc2_preds
            anchors_dict[constants.KEY_CLASSES] = cls_preds
            anchors_dict[constants.KEY_OBJECTNESS] = os_preds
            # anchors_dict[constants.KEY_DIMS] = dim_preds
            # anchors_dict[constants.KEY_ORIENTS_V2] = orient_preds
            anchors_dict[constants.KEY_CORNERS_3D_GRNET] = corners_preds

            # anchors_dict[constants.KEY_FINAL_PROBS] = final_probs

            gt_dict = {}
            gt_dict[constants.KEY_PRIMARY] = feed_dict[
                constants.KEY_LABEL_BOXES_2D]
            gt_dict[constants.KEY_CLASSES] = None
            gt_dict[constants.KEY_BOXES_2D] = None
            gt_dict[constants.KEY_OBJECTNESS] = None
            gt_dict[constants.KEY_BOXES_2D_REFINE] = None
            # gt_dict[constants.KEY_ORIENTS_V2] = None
            # gt_dict[constants.KEY_DIMS] = None
            gt_dict[constants.KEY_CORNERS_3D_GRNET] = None

            auxiliary_dict = {}
            auxiliary_dict[constants.KEY_BOXES_2D] = feed_dict[
                constants.KEY_LABEL_BOXES_2D]
            auxiliary_dict[constants.KEY_STEREO_CALIB_P2] = feed_dict[
                constants.KEY_STEREO_CALIB_P2]
            auxiliary_dict[constants.KEY_BOXES_3D] = feed_dict[
                constants.KEY_LABEL_BOXES_3D]
            gt_labels = feed_dict[constants.KEY_LABEL_CLASSES]
            auxiliary_dict[constants.KEY_CLASSES] = gt_labels
            auxiliary_dict[constants.KEY_NUM_INSTANCES] = feed_dict[
                constants.KEY_NUM_INSTANCES]
            auxiliary_dict[constants.KEY_PROPOSALS] = anchors
            auxiliary_dict[constants.KEY_MEAN_DIMS] = feed_dict[
                constants.KEY_MEAN_DIMS]
            auxiliary_dict[constants.KEY_IMAGE_INFO] = feed_dict[
                constants.KEY_IMAGE_INFO]

            proposals_dict, targets, stats = self.target_generators.generate_targets(
                anchors_dict, gt_dict, auxiliary_dict, subsample=False)

            # recall
            anchors_dict[constants.KEY_PRIMARY] = proposals
            _, _, second_stage_stats = self.target_generators.generate_targets(
                anchors_dict, gt_dict, auxiliary_dict, subsample=False)

            # precision
            fg_probs, _ = final_probs[:, :, 1:].max(dim=-1)
            fake_match = auxiliary_dict[constants.KEY_FAKE_MATCH]
            second_stage_stats.update(
                Analyzer.analyze_precision(
                    fake_match,
                    fg_probs,
                    feed_dict[constants.KEY_NUM_INSTANCES],
                    thresh=0.3))

            prediction_dict[constants.KEY_STATS] = [stats, second_stage_stats]
            prediction_dict[constants.KEY_TARGETS] = targets
            prediction_dict[constants.KEY_PROPOSALS] = anchors
        else:

            prediction_dict[constants.KEY_CLASSES] = final_probs
            # prediction_dict[constants.KEY_OBJECTNESS] = os_preds
            # prediction_dict[constants.KEY_ORIENTS_V2] = decoded_orient_preds
            # prediction_dict[constants.KEY_DIMS] = decoded_dim_preds

            image_info = feed_dict[constants.KEY_IMAGE_INFO]
            proposals[:, :, ::2] = proposals[:, :, ::
                                             2] / image_info[:, 3].unsqueeze(
                                                 -1).unsqueeze(-1)
            proposals[:, :, 1::2] = proposals[:, :, 1::
                                              2] / image_info[:, 2].unsqueeze(
                                                  -1).unsqueeze(-1)
            prediction_dict[constants.KEY_BOXES_2D] = proposals
            prediction_dict['rcnn_3d'] = torch.ones_like(proposals)

            corners_preds = coder.decode_batch(
                corners_preds.detach(), proposals,
                feed_dict[constants.KEY_STEREO_CALIB_P2])
            prediction_dict[constants.KEY_CORNERS_2D] = corners_preds

        if self.training:
            loss_dict = self.loss(prediction_dict, feed_dict)
            return prediction_dict, loss_dict
        else:
            return prediction_dict

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
        loss_dict = {}

        targets = prediction_dict[constants.KEY_TARGETS]

        cls_target = targets[constants.KEY_CLASSES]
        loc1_target = targets[constants.KEY_BOXES_2D]
        loc2_target = targets[constants.KEY_BOXES_2D_REFINE]
        os_target = targets[constants.KEY_OBJECTNESS]
        corners_target = targets[constants.KEY_CORNERS_3D_GRNET]
        # dims_target = targets[constants.KEY_DIMS]
        # orients_target = targets[constants.KEY_ORIENTS_V2]

        loc1_preds = loc1_target['pred']
        loc2_preds = loc2_target['pred']
        loc1_target = loc1_target['target']
        loc2_target = loc2_target['target']
        assert loc1_target.shape == loc2_target.shape
        loc_target = loc1_target

        conf_preds = cls_target['pred']
        conf_target = cls_target['target']
        conf_weight = cls_target['weight']
        conf_target[conf_weight == 0] = -1

        os_preds = os_target['pred']
        os_target_ = os_target['target']
        os_weight = os_target['weight']
        os_target_[os_weight == 0] = -1

        loc_loss, os_loss, conf_loss = self.two_step_loss(
            loc1_preds,
            loc2_preds,
            loc_target,
            conf_preds,
            conf_target,
            os_preds,
            os_target_,
            is_print=False)

        # import ipdb
        # ipdb.set_trace()
        # 3d loss
        # corners_loss = common_loss.calc_loss(self.rcnn_corners_loss,
        # corners_2d_target)

        # import ipdb
        # ipdb.set_trace()
        preds = corners_target['pred']
        targets = corners_target['target']
        weights = corners_target['weight']
        proposals = prediction_dict[constants.KEY_PROPOSALS]
        p2 = feed_dict[constants.KEY_STEREO_CALIB_P2]
        image_info = feed_dict[constants.KEY_IMAGE_INFO]
        weights = weights.unsqueeze(-1)

        local_corners_gt = targets[:, :, :24]
        location_gt = targets[:, :, 24:27]
        dims_gt = targets[:, :, 27:]
        N, M = local_corners_gt.shape[:2]

        global_corners_gt = (local_corners_gt.view(N, M, 8, 3) +
                             location_gt.view(N, M, 1, 3)).view(N, M, -1)
        center_depth_gt = location_gt[:, :, 2:]

        mean_dims = torch.tensor([1.8, 1.8, 3.7]).type_as(preds)
        dims_preds = torch.exp(preds[:, :, :3]) * mean_dims
        # import ipdb
        # ipdb.set_trace()
        dims_loss = self.l1_loss(dims_preds, dims_gt) * weights
        ry_preds = preds[:, :, 3:4]
        # ray_angle = -torch.atan2(location_gt[:, :, 2],
        # location_gt[:, :, 0])
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
                    center_2d_preds[batch_ind], center_depth_preds[batch_ind],
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

        # import ipdb
        # ipdb.set_trace()
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
        # import ipdb
        # ipdb.set_trace()
        corners_2d_loss = self.l1_loss(
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

        # rpn_orients_loss = common_loss.calc_loss(self.rcnn_orient_loss,
        # corners_2d_target) * 100

        # loss

        # import ipdb
        # ipdb.set_trace()
        # loss_dict['total_loss'] = total_loss
        pos = weights > 0  # [N,#anchors]
        num_pos = pos.data.long().sum().clamp(min=1).float()

        loss_dict['loc_loss'] = loc_loss
        loss_dict['os_loss'] = os_loss
        loss_dict['conf_loss'] = conf_loss
        # loss_dict['corners_2d_loss'] = corners_2d_loss.sum() / num_pos * 0.1
        loss_dict['dims_loss'] = dims_loss.sum() / num_pos * 10
        loss_dict['global_corners_loss'] = global_corners_loss.sum() / num_pos * 10
        loss_dict['location_loss'] = location_loss.sum() / num_pos * 10
        loss_dict['center_depth_loss'] = center_depth_loss.sum() / num_pos * 10
        # loss_dict['orients_loss'] = rpn_orients_loss

        return loss_dict

    def loss_orig(self, prediction_dict, feed_dict):
        # loss for cls
        loss_dict = {}

        targets = prediction_dict[constants.KEY_TARGETS]
        cls_target = targets[constants.KEY_CLASSES]
        loc1_target = targets[constants.KEY_BOXES_2D]
        loc2_target = targets[constants.KEY_BOXES_2D_REFINE]
        os_target = targets[constants.KEY_OBJECTNESS]
        dims_target = targets[constants.KEY_DIMS]
        orients_target = targets[constants.KEY_ORIENTS_V2]

        rpn_cls_loss = common_loss.calc_loss(
            focal_loss_alt(self.num_classes), cls_target, normalize=False)
        rpn_loc1_loss = common_loss.calc_loss(self.rpn_bbox_loss, loc1_target)
        rpn_os_loss = common_loss.calc_loss(
            focal_loss_alt(2), os_target, normalize=False)
        rpn_loc2_loss = common_loss.calc_loss(self.rpn_bbox_loss, loc2_target)

        rpn_dims_loss = common_loss.calc_loss(self.rpn_bbox_loss, dims_target)
        rpn_orients_loss = common_loss.calc_loss(self.rcnn_orient_loss,
                                                 orients_target)

        cls_targets = cls_target['target']
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum().clamp(min=1).float()

        cls_loss = rpn_cls_loss / num_pos

        os_loss = rpn_os_loss / num_pos

        loss_dict.update({
            'rpn_cls_loss': cls_loss,
            'rpn_loc1_loss': rpn_loc1_loss * 0.35,
            'rpn_loc2_loss': rpn_loc2_loss * 0.5,
            'rpn_os_loss': os_loss * 10,
            'rpn_dims_loss': rpn_dims_loss,
            'rpn_orients_loss': rpn_orients_loss
        })

        return loss_dict

    def loss_retina(self, prediction_dict, feed_dict):
        loss_dict = {}

        targets = prediction_dict[constants.KEY_TARGETS]
        cls_target = targets[constants.KEY_CLASSES]
        loc1_target = targets[constants.KEY_BOXES_2D]
        loc2_target = targets[constants.KEY_BOXES_2D_REFINE]
        os_target = targets[constants.KEY_OBJECTNESS]

        conf_weight = cls_target['weight']
        conf_target = cls_target['target']
        conf_target[conf_weight == 0] = -1

        os_preds = os_target['pred']
        os_target_ = os_target['target']
        os_weight = os_target['weight']
        os_target_[os_weight == 0] = -1

        total_loss = self.retina_loss(
            loc1_target['pred'], loc2_target['pred'], loc1_target['target'],
            cls_target['pred'], conf_target, os_preds, os_target_)
        loss_dict['total_loss'] = total_loss
        return loss_dict
