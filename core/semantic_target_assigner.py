#!/usr/bin/env python
# encoding: utf-8

import torch

# core classes

from core.analyzer import Analyzer

# builder
from builder import matcher_builder
from builder import bbox_coder_builder
from builder import similarity_calc_builder


class SemanticTargetAssigner(object):
    def __init__(self, assigner_config):

        # some compositions
        self.similarity_calc = similarity_calc_builder.build(
            assigner_config['similarity_calc_config'])
        self.bbox_coder = bbox_coder_builder.build(
            assigner_config['coder_config'])
        self.matcher = matcher_builder.build(assigner_config['matcher_config'])
        self.analyzer = Analyzer()

        self.fg_thresh = assigner_config['fg_thresh']
        self.bg_thresh = assigner_config['bg_thresh']
        # self.clobber_positives = assigner_config['clobber_positives']

    @property
    def stat(self):
        return self.analyzer.stat

    def assign(self, bboxes, gt_boxes, gt_labels=None, cls_prob=None):
        """
        Assign each bboxes with label and bbox targets for training

        Args:
        bboxes: shape(N,K,4), encoded by xxyy
        gt_boxes: shape(N,M,4), encoded likes as bboxes
        """
        # import ipdb
        # ipdb.set_trace()

        # usually IoU overlaps is used as metric
        bboxes = bboxes.detach()
        match_quality_matrix = self.similarity_calc.compare_batch(bboxes,
                                                                  gt_boxes)

        # match
        # shape(N,K)
        match = self.matcher.match_batch(match_quality_matrix, self.fg_thresh)

        self.analyzer.analyze(match, gt_boxes.shape[1])

        # get assigned infomation
        # shape (num_batch,num_boxes)
        assigned_overlaps_batch = self.matcher.assigned_overlaps_batch

        # assign regression targets
        reg_targets = self._assign_regression_targets(match, bboxes, gt_boxes)

        # assign classification targets
        cls_targets = self._assign_classification_targets(match, gt_labels)

        # create regression weights
        reg_weights = self._create_regression_weights(assigned_overlaps_batch)

        # create classification weights
        #  cls_weights = self._create_classification_weights(
        #  assigned_overlaps_batch)
        cls_weights = reg_weights.clone()

        ####################################
        # postprocess
        ####################################
        # match == -1 means unmatched
        reg_targets[match == -1] = 0
        cls_targets[match == -1] = 0
        reg_weights[match == -1] = 0

        # as for cls weights, ignore according to bg_thresh
        if self.bg_thresh > 0:
            ignored_bg = (assigned_overlaps_batch > self.bg_thresh) & (
                match == -1)
            cls_weights[ignored_bg] = 0

        return cls_targets, reg_targets, cls_weights, reg_weights

    def _create_regression_weights(self, assigned_overlaps_batch):
        """
        Args:
        assigned_overlaps_batch: shape (num_batch,num_boxes)
        Returns:
        reg_weights: shape(num_batch,num_boxes,4)
        """
        #  gamma = 2
        #  return torch.pow(1 - assigned_overlaps_batch, gamma).detach()
        #  import ipdb
        #  ipdb.set_trace()
        #  reg_weights = torch.empty_like(assigned_overlaps_batch)
        #  sum_batch = assigned_overlaps_batch.sum()
        #  reg_weights_0 = assigned_overlaps_batch[assigned_overlaps_batch <
        #  0.5].sum() / sum_batch
        #  reg_weights_1 = assigned_overlaps_batch[(
        #  assigned_overlaps_batch >= 0.5) & (assigned_overlaps_batch < 0.6
        #  )].sum() / sum_batch
        #  reg_weights_2 = assigned_overlaps_batch[(
        #  assigned_overlaps_batch >= 0.6) & (assigned_overlaps_batch < 0.7
        #  )].sum() / sum_batch
        #  reg_weights_3 = assigned_overlaps_batch[assigned_overlaps_batch >=
        #  0.7].sum() / sum_batch

        #  if reg_weights_0:
        #  reg_weights_0 = 1 / reg_weights_0
        #  if reg_weights_1:
        #  reg_weights_1 = 1 / reg_weights_1
        #  if reg_weights_2:
        #  reg_weights_2 = 1 / reg_weights_2
        #  if reg_weights_3:
        #  reg_weights_3 = 1 / reg_weights_3

        #  reg_weights.fill_(reg_weights_0)
        #  reg_weights[assigned_overlaps_batch > 0.5] = reg_weights_1
        #  reg_weights[assigned_overlaps_batch > 0.6] = reg_weights_2
        #  reg_weights[assigned_overlaps_batch > 0.7] = reg_weights_3

        # import ipdb
        # ipdb.set_trace()
        reg_weights = torch.ones_like(assigned_overlaps_batch)
        num_0 = torch.nonzero(assigned_overlaps_batch < 0.5).numel()
        num_1 = torch.nonzero((assigned_overlaps_batch >= 0.5) & (
            assigned_overlaps_batch < 0.6)).numel()
        num_2 = torch.nonzero((assigned_overlaps_batch >= 0.6) & (
            assigned_overlaps_batch < 0.7)).numel()
        num_3 = torch.nonzero(assigned_overlaps_batch >= 0.7).numel()
        reg_weights /= num_0
        if num_1:
            reg_weights[assigned_overlaps_batch > 0.5] = 1 / num_1
        if num_2:
            reg_weights[assigned_overlaps_batch > 0.6] = 1 / num_2
        if num_3:
            reg_weights[assigned_overlaps_batch > 0.7] = 1 / num_3
        return reg_weights

    def _create_classification_weights(self, assigned_overlaps_batch):
        """
        All samples can be used for calculating loss,So reserve all.
        """
        #  cls_weights = torch.ones_like(assigned_overlaps_batch)
        #  return cls_weights
        cls_weights = torch.ones_like(assigned_overlaps_batch)
        cls_weights[assigned_overlaps_batch > 0.5] = 2
        cls_weights[assigned_overlaps_batch > 0.6] = 3
        cls_weights[assigned_overlaps_batch > 0.7] = 4
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
        # no need grad_fn
        return reg_targets_batch

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
