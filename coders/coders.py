# -*- coding: utf-8 -*-

from utils.registry import TARGET_ASSIGNERS
from core import constants
import torch
import bbox_coders


class TargetAssigner(object):
    def generate_assigned_label(cls, match, gt_boxes, device='cuda'):
        """
        Args:
            gt_boxes: shape(N, M, K) or shape(N, M)
        """
        is_cls = len(gt_boxes.shape) == 2
        if is_cls:
            gt_boxes = gt_boxes.unsqueeze(-1)
        N, M, K = gt_boxes.shape
        offset = torch.arange(0, N, device=device) * M
        match = match + offset.view(N, 1).type_as(match)
        assigned_gt_boxes = gt_boxes.view(-1, K)[match.view(-1)].view(N, -1, K)
        if is_cls:
            assigned_gt_boxes = assigned_gt_boxes.squeeze(-1)
        return assigned_gt_boxes

    # @classmethod
    # def suppress_ignored_case(match, num_instances):
    # """
    # create new match that ignore some cases
    # Args:
    # match: shape(N, M)
    # num_instances: shape(N, )
    # """

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
    def assign_target(cls, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        gt = kwargs[constants.KEY_CLASSES]
        assigned_gt = cls.generate_assigned_label(
            cls, kwargs[constants.KEY_IGNORED_MATCH], gt)
        assigned_gt[match == -1] = 0

        return assigned_gt.long()

    @classmethod
    def assign_weight(cls, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        assigned_overlaps_batch = kwargs[constants.KEY_ASSIGNED_OVERLAPS]
        bg_thresh = kwargs[constants.KEY_BG_THRESH]
        cls_weights = super().assign_weight(cls, match)
        if bg_thresh > 0:
            ignored_bg = (assigned_overlaps_batch > bg_thresh) & (match == -1)
            cls_weights[ignored_bg] = 0
        return cls_weights


@TARGET_ASSIGNERS.register(constants.KEY_BOXES_2D)
class Box2DTargetAssigner(TargetAssigner):
    @classmethod
    def assign_target(cls, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        gt = kwargs[constants.KEY_BOXES_2D]
        proposals = kwargs[constants.KEY_PROPOSALS]
        assigned_gt = cls.generate_assigned_label(
            cls, kwargs[constants.KEY_IGNORED_MATCH], gt)
        # prepare coder
        coder = bbox_coders.build({'type': constants.KEY_BOXES_2D})
        reg_targets_batch = coder.encode_batch(proposals, assigned_gt)
        reg_targets_batch[match == -1] = 0
        # no need grad_fn
        return reg_targets_batch

    @classmethod
    def assign_weight(cls, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        reg_weights = super().assign_weight(cls, match)
        reg_weights[match == -1] = 0
        return reg_weights


@TARGET_ASSIGNERS.register(constants.KEY_ORIENTS)
class OrientsTargetAssigner(TargetAssigner):
    @classmethod
    def assign_target(cls, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        gt = kwargs[constants.KEY_BOXES_3D]
        assigned_gt = cls.generate_assigned_label(
            cls, kwargs[constants.KEY_IGNORED_MATCH], gt)
        proposals = kwargs[constants.KEY_PROPOSALS]
        p2 = kwargs[constants.KEY_STEREO_CALIB_P2]

        coder = bbox_coders.build({'type': constants.KEY_ORIENTS})
        reg_targets_batch = coder.encode_batch(assigned_gt, proposals, p2)
        reg_targets_batch[match == -1] = 0
        # no need grad_fn
        return reg_targets_batch

    @classmethod
    def assign_weight(cls, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        reg_weights = super().assign_weight(cls, match)
        reg_weights[match == -1] = 0
        return reg_weights


# multibin
@TARGET_ASSIGNERS.register(constants.KEY_ORIENTS_V3)
class OrientsV3TargetAssigner(TargetAssigner):
    @classmethod
    def assign_target(cls, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        gt = kwargs[constants.KEY_BOXES_3D]
        assigned_gt = cls.generate_assigned_label(
            cls, kwargs[constants.KEY_IGNORED_MATCH], gt)

        coder = bbox_coders.build({'type': constants.KEY_ORIENTS_V3})
        reg_targets_batch = coder.encode_batch(assigned_gt)
        reg_targets_batch[match == -1] = 0
        # no need grad_fn
        return reg_targets_batch

    @classmethod
    def assign_weight(cls, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        reg_weights = super().assign_weight(cls, match)
        reg_weights[match == -1] = 0
        return reg_weights


@TARGET_ASSIGNERS.register(constants.KEY_DIMS)
class DimsTargetAssigner(TargetAssigner):
    @classmethod
    def assign_target(cls, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        gt = kwargs[constants.KEY_BOXES_3D][:, :, 3:6]
        label_classes = kwargs[constants.KEY_CLASSES]
        mean_dims = kwargs[constants.KEY_MEAN_DIMS]
        bg_dim = torch.zeros_like(mean_dims[:, -1:, :])
        mean_dims = torch.cat([bg_dim, mean_dims], dim=1)
        mean_dims = cls.generate_assigned_label(cls,
                                                label_classes.long(),
                                                mean_dims)

        # prepare coder
        coder = bbox_coders.build({'type': constants.KEY_DIMS})
        reg_targets_batch = coder.encode_batch(gt, mean_dims)

        reg_targets_batch = cls.generate_assigned_label(
            cls, kwargs[constants.KEY_IGNORED_MATCH], reg_targets_batch)

        reg_targets_batch[match == -1] = 0
        # no need grad_fn
        return reg_targets_batch

    @classmethod
    def assign_weight(cls, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        reg_weights = super().assign_weight(cls, match)
        reg_weights[match == -1] = 0
        return reg_weights
