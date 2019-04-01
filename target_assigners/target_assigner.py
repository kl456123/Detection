#!/usr/bin/env python
# encoding: utf-8

import torch

# core classes

from utils.registry import TARGET_ASSIGNERS
import similarity_calcs
import bbox_coders
import matchers
from core import constants


@TARGET_ASSIGNERS.register('faster_rcnn')
class TargetAssigner(object):
    def __init__(self, assigner_config):

        # some compositions
        self.similarity_calc = similarity_calcs.build(
            assigner_config['similarity_calc_config'])
        self.bbox_coder = bbox_coders.build(assigner_config['coder_config'])
        self.matcher = matchers.build(assigner_config['matcher_config'])

        self.fg_thresh = assigner_config['fg_thresh']
        self.bg_thresh = assigner_config['bg_thresh']
    def suppress_ignore(self, match_quality_matrix, num_instances):
        """
        Args:
            match_quality_matrix: shape(N, M, K)
            num_instances: shape(N, ), it determines the num of valid instances,
            it refers to the last dim of match_quality_matrix
        """
        N, M, K = match_quality_matrix.shape
        num_instances = num_instances.unsqueeze(-1).repeat(1, M)
        offsets = torch.arange(0, ) * K
        num_instances = offsets + num_instances
        match_quality_matrix.view(-1, K)

    def assign(self, proposals_dict, gt_dict, num_instances, device='cuda'):
        """
        Assign each bboxes with label and bbox targets for training

        Args:
            bboxes: shape(N,K,4), encoded by xxyy
            gt_boxes: shape(N,M,4), encoded likes as bboxes
        """
        # usually IoU overlaps is used as metric
        proposals_primary = proposals_dict[constants.KEY_PRIMARY].detach()
        gt_primary = gt_dict[constants.KEY_PRIMARY].detach()

        match_quality_matrix = self.similarity_calc.compare_batch(
            proposals_primary, gt_primary)



        match = self.matcher.match_batch(match_quality_matrix, self.fg_thresh)
        assigned_overlaps_batch = self.matcher.assigned_overlaps_batch.to(
            device)

        # assign regression targets
        reg_targets = self._assign_regression_targets(match, proposals_primary,
                                                      gt_primary)

        gt_labels = gt_dict[constants.KEY_CLASSES]
        # assign classification targets
        cls_targets = self._assign_classification_targets(match, gt_labels)

        # create regression weights
        reg_weights = self._create_regression_weights(assigned_overlaps_batch)

        # create classification weights
        cls_weights = self._create_classification_weights(
            assigned_overlaps_batch)

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

        targets = {}
        targets[constants.KEY_CLASSES] = {
            'weight': cls_weights,
            'target': cls_targets
        }
        targets[constants.KEY_BOXES_2D] = {
            'weight': reg_weights,
            'target': reg_targets
        }
        return targets

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

    def _assign_regression_targets(self,
                                   match,
                                   bboxes,
                                   gt_boxes,
                                   device='cuda'):
        """
        Args:
            match: Tensor(num_batch,num_boxes)
            gt_boxes: Tensor(num_batch,num_gt_boxes,4)
        Returns:
            reg_targets: Tensor(num_batch,num_boxes,4)
        """
        # shape(num_batch,num_boxes,4)
        batch_size = gt_boxes.shape[0]
        offset = torch.arange(0, batch_size, device=device) * gt_boxes.size(1)
        match = match + offset.view(batch_size, 1).type_as(match)
        assigned_gt_boxes = gt_boxes.view(-1, 4)[match.view(-1)].view(
            batch_size, -1, 4)
        reg_targets_batch = self.bbox_coder.encode_batch(bboxes,
                                                         assigned_gt_boxes)
        # no need grad_fn
        return reg_targets_batch

    def _assign_classification_targets(self, match, gt_labels, device='cuda'):
        """
        Just return the countpart labels
        Note that use zero to represent background labels
        For the first stage, generate binary labels, For the second stage
        generate countpart gt_labels
        """
        # multiple labels classification
        batch_size = match.shape[0]
        offset = torch.arange(0, batch_size, device=device) * gt_labels.size(1)
        match = match + offset.view(batch_size, 1).type_as(match)
        cls_targets_batch = gt_labels.view(-1)[match.view(-1)].view(
            batch_size, match.shape[1])
        return cls_targets_batch.long()

    def _generate_binary_labels(self, match, device='cuda'):
        gt_labels_batch = torch.ones_like(match, device=device).long()
        return gt_labels_batch
