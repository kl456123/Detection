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
import anchor_generators
from utils.registry import DETECTORS
import bbox_coders
import torch.nn.functional as F
from core.utils.analyzer import Analyzer


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


@DETECTORS.register('prnet')
class TwoStageRetinaLayer(Model):
    def init_param(self, model_config):
        # including bg
        self.num_classes = len(model_config['classes']) + 1
        self.in_channels = model_config.get('in_channels', 128)
        self.num_regress = model_config.get('num_regress', 4)
        self.feature_extractor_config = model_config['feature_extractor_config']

        self.target_generators = TargetGenerator(
            model_config['target_generator_config'])
        self.anchor_generator = anchor_generators.build(
            model_config['anchor_generator_config'])

        self.num_anchors = self.anchor_generator.num_anchors
        input_size = torch.tensor(model_config['input_size']).float()

        self.normlize_anchor = False
        self.anchors = self.anchor_generator.generate(
            input_size, normalize=self.normlize_anchor)

        self.use_focal_loss = model_config['use_focal_loss']

    def init_modules(self):

        in_channels = self.in_channels

        self.loc_feature1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.loc_feature2 = nn.Sequential(
            nn.Conv2d(
                in_channels * 2, in_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.cls_feature1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.cls_feature2 = nn.Sequential(
            nn.Conv2d(
                in_channels * 2, in_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

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

    def forward(self, feed_dict):
        features = self.feature_extractor(feed_dict[constants.KEY_IMAGE])
        y_locs1 = []
        y_locs2 = []
        y_os = []
        y_cls = []

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

        loc1_preds = torch.cat(y_locs1, dim=1)
        loc2_preds = torch.cat(y_locs2, dim=1)
        os_preds = torch.cat(y_os, dim=1)
        cls_preds = torch.cat(y_cls, dim=1)

        # if self.training:
        # prediction_dict = {
        # 'loc1_preds': loc1_preds,
        # 'loc2_preds': loc2_preds,
        # 'os_preds': os_preds,
        # 'cls_preds': cls_preds
        # }

        # stats = {
        # 'recall': torch.tensor([1, 1]).to('cuda').float().unsqueeze(0)
        # }
        # prediction_dict[constants.KEY_STATS] = [stats]
        # else:
        # prediction_dict = {}
        # cls_probs = F.softmax(cls_preds, dim=-1)
        # os_probs = F.softmax(os_preds, dim=-1)[:, :, 1:]
        # os_probs[os_probs <= 0.4] = 0
        # prediction_dict[constants.KEY_CLASSES] = cls_probs * os_probs
        # # prediction_dict[constants.KEY_OBJECTNESS] = os_preds

        # image_info = feed_dict[constants.KEY_IMAGE_INFO]
        # variances = [0.1, 0.2]
        # default_boxes = feed_dict['default_boxes'][0]
        # new_default_boxes = torch.cat([
        # default_boxes[:, :2] - default_boxes[:, 2:] / 2,
        # default_boxes[:, :2] + default_boxes[:, 2:] / 2
        # ], 1)
        # xymin = loc2_preds[0, :, :2] * variances[
        # 0] * default_boxes[:, 2:] + new_default_boxes[:, :2]
        # xymax = loc2_preds[0, :, 2:] * variances[
        # 0] * default_boxes[:, 2:] + new_default_boxes[:, 2:]
        # proposals = torch.cat([xymin, xymax], 1).unsqueeze(0)  # [8732,4]

        # image_info = image_info.unsqueeze(-1).unsqueeze(-1)
        # proposals[:, :, ::
        # 2] = proposals[:, :, ::
        # 2] * image_info[:, 1] / image_info[:, 3]
        # proposals[:, :, 1::
        # 2] = proposals[:, :, 1::
        # 2] * image_info[:, 0] / image_info[:, 2]
        # prediction_dict[constants.KEY_BOXES_2D] = proposals
        # return prediction_dict

        image_info = feed_dict[constants.KEY_IMAGE_INFO]

        batch_size = loc1_preds.shape[0]

        anchors = self.anchors.cuda()
        anchors = anchors.repeat(batch_size, 1, 1)

        coder = bbox_coders.build(
            self.target_generators.target_generator_config['coder_config'])
        proposals = coder.decode_batch(loc2_preds, anchors).detach()

        # if self.normlize_anchor:
        # denormalize
        # h = image_info[:, 0].unsqueeze(-1).unsqueeze(-1)
        # w = image_info[:, 1].unsqueeze(-1).unsqueeze(-1)
        # proposals[:, :, ::2] = proposals[:, :, ::2] * w
        # proposals[:, :, 1::2] = proposals[:, :, 1::2] * h

        cls_probs = F.softmax(cls_preds.detach(), dim=-1)
        os_probs = F.softmax(os_preds.detach(), dim=-1)[:, :, 1:]
        os_probs[os_probs <= 0.4] = 0
        # final_probs = cls_probs * os_probs
        # import ipdb
        # ipdb.set_trace()
        final_probs = cls_probs * os_probs
        image_info = feed_dict[constants.KEY_IMAGE_INFO].unsqueeze(
            -1).unsqueeze(-1)

        prediction_dict = {}
        if self.training:

            # anchors = prediction_dict['anchors']
            anchors_dict = {}
            anchors_dict[constants.KEY_PRIMARY] = anchors
            anchors_dict[constants.KEY_BOXES_2D] = loc1_preds
            anchors_dict[constants.KEY_BOXES_2D_REFINE] = loc2_preds
            anchors_dict[constants.KEY_CLASSES] = cls_preds
            anchors_dict[constants.KEY_OBJECTNESS] = os_preds
            # anchors_dict[constants.KEY_FINAL_PROBS] = final_probs

            gt_dict = {}
            gt_dict[constants.KEY_PRIMARY] = feed_dict[constants.
                                                       KEY_LABEL_BOXES_2D]
            gt_dict[constants.KEY_CLASSES] = None
            gt_dict[constants.KEY_BOXES_2D] = None
            gt_dict[constants.KEY_OBJECTNESS] = None
            gt_dict[constants.KEY_BOXES_2D_REFINE] = None

            auxiliary_dict = {}
            label_boxes_2d = feed_dict[constants.KEY_LABEL_BOXES_2D]
            if self.normlize_anchor:
                label_boxes_2d[:, :, ::2] = label_boxes_2d[:, :, ::
                                                           2] / image_info[:, 1]
                label_boxes_2d[:, :, 1::2] = label_boxes_2d[:, :, 1::
                                                            2] / image_info[:,
                                                                            0]
            auxiliary_dict[constants.KEY_BOXES_2D] = label_boxes_2d
            gt_labels = feed_dict[constants.KEY_LABEL_CLASSES]
            auxiliary_dict[constants.KEY_CLASSES] = gt_labels
            auxiliary_dict[constants.KEY_NUM_INSTANCES] = feed_dict[
                constants.KEY_NUM_INSTANCES]
            auxiliary_dict[constants.KEY_PROPOSALS] = anchors

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
        else:

            prediction_dict[constants.KEY_CLASSES] = final_probs
            # prediction_dict[constants.KEY_OBJECTNESS] = os_preds

            proposals[:, :, ::2] = proposals[:, :, ::2] / image_info[:, 3]
            proposals[:, :, 1::2] = proposals[:, :, 1::2] / image_info[:, 2]
            prediction_dict[constants.KEY_BOXES_2D] = proposals
        return prediction_dict

    # def loss(self, prediction_dict, feed_dict):
    # # import ipdb
    # # ipdb.set_trace()

    # target = feed_dict['gt_target']
    # loc1_preds = prediction_dict['loc1_preds']
    # loc2_preds = prediction_dict['loc2_preds']
    # conf_preds = prediction_dict['cls_preds']
    # os_preds = prediction_dict['os_preds']
    # bbox, labels, os_gt, _ = target

    # loc_loss, os_loss, conf_loss = self.two_step_loss(
    # loc1_preds,
    # loc2_preds,
    # bbox,
    # conf_preds,
    # labels.long(),
    # os_preds,
    # os_gt,
    # is_print=False)

    # # loss
    # loss_dict = {}

    # # loss_dict['total_loss'] = total_loss
    # loss_dict['loc_loss'] = loc_loss
    # loss_dict['os_loss'] = os_loss
    # loss_dict['conf_loss'] = conf_loss

    # return loss_dict

    def loss(self, prediction_dict, feed_dict):
        loss_dict = {}

        targets = prediction_dict[constants.KEY_TARGETS]

        cls_target = targets[constants.KEY_CLASSES]
        loc1_target = targets[constants.KEY_BOXES_2D]
        loc2_target = targets[constants.KEY_BOXES_2D_REFINE]
        os_target = targets[constants.KEY_OBJECTNESS]

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

        # loss

        # loss_dict['total_loss'] = total_loss
        loss_dict['loc_loss'] = loc_loss
        loss_dict['os_loss'] = os_loss
        loss_dict['conf_loss'] = conf_loss

        return loss_dict

    def loss_orig(self, prediction_dict, feed_dict):
        # loss for cls
        loss_dict = {}

        targets = prediction_dict[constants.KEY_TARGETS]
        cls_target = targets[constants.KEY_CLASSES]
        loc1_target = targets[constants.KEY_BOXES_2D]
        loc2_target = targets[constants.KEY_BOXES_2D_REFINE]
        os_target = targets[constants.KEY_OBJECTNESS]

        rpn_cls_loss = common_loss.calc_loss(
            focal_loss_alt(self.num_classes), cls_target, normalize=False)
        rpn_loc1_loss = common_loss.calc_loss(self.rpn_bbox_loss, loc1_target)
        rpn_os_loss = common_loss.calc_loss(
            focal_loss_alt(2), os_target, normalize=False)
        rpn_loc2_loss = common_loss.calc_loss(self.rpn_bbox_loss, loc2_target)

        cls_targets = cls_target['target']
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum().clamp(min=1).float()

        cls_loss = rpn_cls_loss / num_pos

        os_loss = rpn_os_loss / num_pos

        loss_dict.update({
            'rpn_cls_loss': cls_loss,
            'rpn_loc1_loss': rpn_loc1_loss * 0.35,
            'rpn_loc2_loss': rpn_loc2_loss * 0.5,
            'rpn_os_loss': os_loss * 10
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
