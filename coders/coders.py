# -*- coding: utf-8 -*-

from utils.registry import TARGET_ASSIGNERS
from core import constants
import torch
import bbox_coders


class TargetAssigner(object):

    def generate_assigned_label(self, match, gt_boxes, device='cuda'):
        """
        Args:
            gt_boxes: shape(N, M, K) or shape(N, M)
        """
        if len(gt_boxes.shape) == 2:
            gt_boxes = gt_boxes.unsqueeze(-1)
        N, M, K = gt_boxes.shape
        offset = torch.arange(0, N, device=device) * M
        match = match + offset.view(N, 1).type_as(match)
        assigned_gt_boxes = gt_boxes.view(-1, K)[match.view(-1)].view(
            N, -1, K)
        if len(gt_boxes.shape) == 2:
            assigned_gt_boxes = assigned_gt_boxes.squeeze(-1)
        return assigned_gt_boxes

    # def _assign_target(self, assigned_gt, *args, **kwargs):
        # raise NotImplementedError

    # def assign_target(self, match, gt, *args, **kwargs):
        # assigned_gt = self.generate_assigned_label(match, gt)
        # return self._assign_target(assigned_gt, *args, **kwargs)

    def assign_weight(self, match):
        """
        Args:
            match: shape(N, M), -1 refers to no anyone matched
        """
        return torch.ones_like(match).float()


@TARGET_ASSIGNERS.register(constants.KEY_CLASSES)
class ClassesTargetAssigner(TargetAssigner):
    @classmethod
    def assign_target(cls, match, gt):
        assigned_gt = cls.generate_assigned_label(match, gt)
        assigned_gt[match == -1] = 0

        return assigned_gt, None

    @classmethod
    def assign_weight(cls, match, assigned_overlaps_batch, bg_thresh=0):
        cls_weights = super().assign_weight(match)
        if bg_thresh > 0:
            ignored_bg = (assigned_overlaps_batch > bg_thresh) & (
                match == -1)
            cls_weights[ignored_bg] = 0
        return cls_weights


@TARGET_ASSIGNERS.register('constants.KEY_BOXES_2D')
class Box2DTargetAssigner(TargetAssigner):

    @classmethod
    def assign_target(cls, match, gt, proposals):
        assigned_gt = cls.generate_assigned_label(match, gt)
        # prepare coder
        coder_config = {'type': 'center',
                        'bbox_normalize_targets_precomputed': False}
        coder = bbox_coders.build(coder_config)
        reg_targets_batch = coder.encode_batch(proposals,
                                               assigned_gt)
        reg_targets_batch[match == -1] = 0
        # no need grad_fn
        return reg_targets_batch, coder

    @classmethod
    def assign_weight(cls, match):
        reg_weights = super().assign_weight(match)
        reg_weights[match == -1] = 0
        return reg_weights