#!/usr/bin/env python
# encoding: utf-8

import torch

# core classes

from core.analyzer import Analyzer

# builder
from builder import matcher_builder
from builder import bbox_coder_builder
from builder import similarity_calc_builder


class TargetAssigner(object):
    def __init__(self, assigner_config):

        # some compositions
        self.similarity_calc = similarity_calc_builder.build(
            assigner_config['similarity_calc_config'])
        self.bbox_coder = bbox_coder_builder.build(
            assigner_config['coder_config'])
        self.matcher = matcher_builder.build(assigner_config['matcher_config'])
        self.analyzer = Analyzer()

        # cls thresh
        self.fg_thresh_cls = assigner_config['fg_thresh_cls']
        self.bg_thresh_cls = assigner_config['bg_thresh_cls']

        # bbox thresh
        self.fg_thresh_reg = assigner_config['fg_thresh_reg']
        self.bg_thresh_reg = assigner_config['bg_thresh_reg']

        if assigner_config.get('fake_match_thresh') is not None:
            self.fake_match_thresh = assigner_config['fake_match_thresh']
        else:
            # default value
            self.fake_match_thresh = 0.7

    @property
    def stat(self):
        return self.analyzer.stat

    def assign(self,
               bboxes,
               gt_boxes,
               gt_labels=None,
               cls_prob=None,
               ret_iou=False):
        """
        Assign each bboxes with label and bbox targets for training

        Args:
            bboxes: shape(N,K,4), encoded by xxyy
            gt_boxes: shape(N,M,4), encoded likes as bboxes
        """

        # usually IoU overlaps is used as metric
        bboxes = bboxes.detach()
        match_quality_matrix = self.similarity_calc.compare_batch(bboxes,
                                                                  gt_boxes)

        # match 0.7 for truly recall calculation
        # if self.fg_thresh_cls < 0.7:
        fake_match = self.matcher.match_batch(match_quality_matrix,
                                              self.fake_match_thresh)
        stats = self.analyzer.analyze(fake_match, gt_boxes.shape[1])

        #################################
        # handle cls
        #################################
        cls_match = self.matcher.match_batch(match_quality_matrix,
                                             self.fg_thresh_cls)
        cls_assigned_overlaps_batch = self.matcher.assigned_overlaps_batch
        stats['iou'] = cls_assigned_overlaps_batch

        # assign classification targets
        cls_targets = self._assign_classification_targets(cls_match, gt_labels)

        # create classification weights
        cls_weights = self._create_classification_weights(
            cls_assigned_overlaps_batch)

        cls_targets[cls_match == -1] = 0

        # as for cls weights, ignore according to bg_thresh
        if self.bg_thresh_cls > 0:
            ignored_bg = (cls_assigned_overlaps_batch > self.bg_thresh_cls) & (
                cls_match == -1)
            cls_weights[ignored_bg] = 0

        ##################################
        # handle reg
        ##################################
        reg_match = self.matcher.match_batch(match_quality_matrix,
                                             self.fg_thresh_reg)
        reg_assigned_overlaps_batch = self.matcher.assigned_overlaps_batch

        # assign regression targets
        reg_targets, neg_targets_w, neg_targets_h = self._assign_regression_targets(
            reg_match, bboxes, gt_boxes)

        # create regression weights
        reg_weights = self._create_regression_weights(
            reg_assigned_overlaps_batch)

        ###################################
        # handle negative case
        ###################################

        # shape (dx,dy,dw,dw)

        neg_targets = reg_targets[reg_match == -1]
        neg_targets[:, 0] = 0
        neg_targets[:, 1] = 0

        neg_targets[:, 2] = neg_targets_w[reg_match == -1]
        neg_targets[:, 3] = neg_targets_h[reg_match == -1]
        reg_targets[reg_match == -1] = neg_targets

        #  # here shape change to (N,M,4)
        reg_weights = reg_weights.unsqueeze(2).expand_as(
            reg_targets).contiguous()
        reg_weights[:, :, 0] = 0
        reg_weights[:, :, 1] = 0
        reg_weights[reg_match > -1] = 1

        if self.bg_thresh_reg > 0:
            ignored_bg = (reg_assigned_overlaps_batch > self.bg_thresh_reg) & (
                reg_match == -1)
            reg_weights[ignored_bg] = 0

        # reg_weights[reg_match == -1] = 0
        # reg_targets[reg_match == -1] = 0

        ret = [cls_targets, reg_targets, cls_weights, reg_weights, stats]
        if ret_iou:
            ret.append(match_quality_matrix)
        return ret

    #  def generate_neg_targets(self, reg_targets, reg_match):
    #  dw = reg_targets[reg_match == -1, 2]

    def _create_regression_weights(self, assigned_overlaps_batch):
        """
        Args:
        assigned_overlaps_batch: shape (num_batch,num_boxes)
        Returns:
        reg_weights: shape(num_batch,num_boxes,4)
        """
        #  gamma = 2
        #  return torch.pow(1 - assigned_overlaps_batch, gamma).detach()
        return torch.ones_like(assigned_overlaps_batch)

    def _create_classification_weights(self, assigned_overlaps_batch):
        """
        All samples can be used for calculating loss,So reserve all.
        """
        cls_weights = torch.ones_like(assigned_overlaps_batch)
        return cls_weights

    def _assign_regression_targets(self, match, bboxes, gt_boxes):
        """
        Args:
            match: Tensor(num_batch,num_boxes)
            gt_boxes: Tensor(num_batch,num_gt_boxes,4)
        Returns:
            reg_targets: Tensor(num_batch,num_boxes,4)
        """
        # shape(num_batch,num_boxes,4)
        batch_size = gt_boxes.shape[0]
        offset = torch.arange(0, batch_size) * gt_boxes.size(1)
        match += offset.view(batch_size, 1).type_as(match)
        assigned_gt_boxes = gt_boxes.view(-1, 4)[match.view(-1)].view(
            batch_size, -1, 4)
        reg_targets_batch = self.bbox_coder.encode_batch(bboxes,
                                                         assigned_gt_boxes)
        if bboxes.dim() == 2:
            bboxes = bboxes.unsqueeze(0)
        bboxes_w = bboxes[:, :, 2] - bboxes[:, :, 0] + 1
        bboxes_h = bboxes[:, :, 3] - bboxes[:, :, 1] + 1

        neg_targets_w = -torch.log(bboxes_w)
        neg_targets_h = -torch.log(bboxes_h)

        # no need grad_fn
        return reg_targets_batch, neg_targets_w, neg_targets_h

    #  def generate_neg_targets(self, gt_boxes):
    #  """
    #  as for neg case , generate specific target for them
    #  Args:
    #  assigned_gt_boxes: shape(N,4), (x1,y1,x2,y2)
    #  Returns:
    #  neg_targets: shape(N,M,4), (x,y,w,h)
    #  """
    #  center_x = (gt_boxes[:, :, 0] + gt_boxes[:, :, 2]) / 2
    #  center_y = (gt_boxes[:, :, 1] + gt_boxes[:, :, 3]) / 2
    #  w = gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1.0
    #  h = gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1.0

    #  neg_targets = torch.ones_like(gt_boxes)

    def _assign_classification_targets(self, match, gt_labels):
        """
        Just return the countpart labels
        Note that use zero to represent background labels
        For the first stage, generate binary labels, For the second stage
        generate countpart gt_labels
        """
        # binary labels classifcation
        if gt_labels is None:
            # consider it as binary classification problem
            return self._generate_binary_labels(match)

        # multiple labels classification
        batch_size = match.shape[0]
        offset = torch.arange(0, batch_size) * gt_labels.size(1)
        match += offset.view(batch_size, 1).type_as(match)
        cls_targets_batch = gt_labels.view(-1)[match.view(-1)].view(
            batch_size, match.shape[1])
        return cls_targets_batch

    def _generate_binary_labels(self, match):
        gt_labels_batch = torch.ones_like(match).long()
        return gt_labels_batch
