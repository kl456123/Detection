# -*- coding: utf-8 -*-

from utils.registry import TARGET_ASSIGNERS
from core import constants
import torch
import bbox_coders
from utils import geometry_utils


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


class RegTargetAssigner(TargetAssigner):
    @classmethod
    def assign_weight(cls, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        reg_weights = super().assign_weight(cls, match)
        reg_weights[match == -1] = 0
        return reg_weights


@TARGET_ASSIGNERS.register(constants.KEY_CLASSES)
class ClassesTargetAssigner(TargetAssigner):
    @classmethod
    def assign_target(cls, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        gt = kwargs[constants.KEY_CLASSES]
        assigned_gt = cls.generate_assigned_label(
            cls, kwargs[constants.KEY_MATCH], gt)
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


@TARGET_ASSIGNERS.register(constants.KEY_OBJECTNESS)
class ObjectnessTargetAssigner(TargetAssigner):
    @classmethod
    def assign_target(cls, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        gt = torch.ones_like(kwargs[constants.KEY_CLASSES])
        assigned_gt = cls.generate_assigned_label(
            cls, kwargs[constants.KEY_MATCH], gt)
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
@TARGET_ASSIGNERS.register(constants.KEY_BOXES_2D_REFINE)
class Box2DTargetAssigner(RegTargetAssigner):
    @classmethod
    def assign_target(cls, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        gt = kwargs[constants.KEY_BOXES_2D]
        proposals = kwargs[constants.KEY_PROPOSALS]
        assigned_gt = cls.generate_assigned_label(
            cls, kwargs[constants.KEY_MATCH], gt)
        # prepare coder
        # 2d coder config
        bbox_coder_config = kwargs[constants.KEY_TARGET_GENERATOR_CONFIG][
            'coder_config']
        coder = bbox_coders.build(bbox_coder_config)
        reg_targets_batch = coder.encode_batch(proposals, assigned_gt)
        reg_targets_batch[match == -1] = 0
        # no need grad_fn
        return reg_targets_batch


@TARGET_ASSIGNERS.register(constants.KEY_ORIENTS)
class OrientsTargetAssigner(RegTargetAssigner):
    @classmethod
    def assign_target(cls, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        gt = kwargs[constants.KEY_BOXES_3D]
        assigned_gt = cls.generate_assigned_label(
            cls, kwargs[constants.KEY_MATCH], gt)
        proposals = kwargs[constants.KEY_PROPOSALS]
        p2 = kwargs[constants.KEY_STEREO_CALIB_P2]

        coder = bbox_coders.build({'type': constants.KEY_ORIENTS})
        reg_targets_batch = coder.encode_batch(assigned_gt, proposals, p2)
        reg_targets_batch[match == -1] = 0
        # no need grad_fn
        return reg_targets_batch


# multibin
@TARGET_ASSIGNERS.register(constants.KEY_ORIENTS_V3)
class OrientsV3TargetAssigner(RegTargetAssigner):
    @classmethod
    def assign_target(cls, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        gt = kwargs[constants.KEY_BOXES_3D]
        assigned_gt = cls.generate_assigned_label(
            cls, kwargs[constants.KEY_MATCH], gt)

        coder = bbox_coders.build({'type': constants.KEY_ORIENTS_V3})
        reg_targets_batch = coder.encode_batch(assigned_gt)
        reg_targets_batch[match == -1] = 0
        # no need grad_fn
        return reg_targets_batch


# visible side estimation
@TARGET_ASSIGNERS.register(constants.KEY_ORIENTS_V2)
class OrientsV2TargetAssigner(RegTargetAssigner):
    @classmethod
    def assign_target(cls, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        gt = kwargs[constants.KEY_BOXES_3D]
        assigned_gt = cls.generate_assigned_label(
            cls, kwargs[constants.KEY_MATCH], gt)
        proposals = kwargs[constants.KEY_PROPOSALS]
        p2 = kwargs[constants.KEY_STEREO_CALIB_P2]

        coder = bbox_coders.build({'type': constants.KEY_ORIENTS_V2})
        reg_targets_batch = coder.encode_batch(assigned_gt, proposals, p2)
        reg_targets_batch[match == -1] = 0
        # no need grad_fn
        return reg_targets_batch


@TARGET_ASSIGNERS.register(constants.KEY_REAR_SIDE)
class RearSideTargetAssigner(RegTargetAssigner):
    @classmethod
    def assign_target(cls, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        gt = kwargs[constants.KEY_BOXES_3D]
        assigned_gt = cls.generate_assigned_label(
            cls, kwargs[constants.KEY_MATCH], gt)
        proposals = kwargs[constants.KEY_PROPOSALS]
        p2 = kwargs[constants.KEY_STEREO_CALIB_P2]

        coder = bbox_coders.build({'type': constants.KEY_REAR_SIDE})
        reg_targets_batch = coder.encode_batch(assigned_gt, proposals, p2)
        reg_targets_batch[match == -1] = 0
        # no need grad_fn
        return reg_targets_batch

    # @classmethod
    # def assign_weight(cls, **kwargs):
    # import ipdb
    # ipdb.set_trace()
    # match = kwargs[constants.KEY_MATCH]
    # valid_cond = cls.get_valid_cond(**kwargs)

    # reg_weights = super().assign_weight(**kwargs)
    # reg_weights[(match == -1) | (~valid_cond)] = 0

    # return reg_weights

    # @classmethod
    # def get_valid_cond(cls, **kwargs):
    # label_boxes_3d = kwargs[constants.KEY_BOXES_3D]
    # p2 = kwargs[constants.KEY_STEREO_CALIB_P2]
    # batch_size = label_boxes_3d.shape[0]
    # valid_cond = []
    # for batch_ind in range(batch_size):
    # corners_3d = geometry_utils.torch_boxes_3d_to_corners_3d(
    # label_boxes_3d[batch_ind])
    # corners_2d = geometry_utils.torch_points_3d_to_points_2d(
    # corners_3d.reshape((-1, 3)), p2[batch_ind]).reshape(-1, 8, 2)

    # # shape(N, 2, 2)
    # left_side = torch.stack(
    # [corners_2d[:, 0], corners_2d[:, 3]], dim=1)
    # right_side = torch.stack(
    # [corners_2d[:, 1], corners_2d[:, 2]], dim=1)
    # left_slope = geometry_utils.torch_line_to_orientation(
    # left_side[:, 0], left_side[:, 1])
    # right_slope = geometry_utils.torch_line_to_orientation(
    # right_side[:, 0], right_side[:, 1])
    # non_visible_cond = left_slope * right_slope < 0
    # valid_cond.append(non_visible_cond)

    # return torch.stack(valid_cond, dim=0)


@TARGET_ASSIGNERS.register(constants.KEY_DIMS)
class DimsTargetAssigner(RegTargetAssigner):
    @classmethod
    def assign_target(cls, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        gt = kwargs[constants.KEY_BOXES_3D][:, :, 3:6]
        label_classes = kwargs[constants.KEY_CLASSES]
        mean_dims = kwargs[constants.KEY_MEAN_DIMS]
        bg_dim = torch.zeros_like(mean_dims[:, -1:, :])
        mean_dims = torch.cat([bg_dim, mean_dims], dim=1)
        mean_dims = cls.generate_assigned_label(cls, label_classes.long(),
                                                mean_dims)

        # prepare coder
        coder = bbox_coders.build({'type': constants.KEY_DIMS})
        reg_targets_batch = coder.encode_batch(gt, mean_dims)

        reg_targets_batch = cls.generate_assigned_label(
            cls, kwargs[constants.KEY_MATCH], reg_targets_batch)

        reg_targets_batch[match == -1] = 0
        # no need grad_fn
        return reg_targets_batch


@TARGET_ASSIGNERS.register(constants.KEY_CORNERS_2D)
class Corners2DTargetAssigner(RegTargetAssigner):
    @classmethod
    def assign_target(cls, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        # label_boxes_2d = kwargs[constants.KEY_BOXES_2D]
        proposals = kwargs[constants.KEY_PROPOSALS]
        label_boxes_3d = kwargs[constants.KEY_BOXES_3D]
        p2 = kwargs[constants.KEY_STEREO_CALIB_P2]
        image_info = kwargs[constants.KEY_IMAGE_INFO]

        # prepare coder
        # 2d coder config
        label_boxes_3d = cls.generate_assigned_label(
            cls, kwargs[constants.KEY_MATCH], label_boxes_3d)
        coder = bbox_coders.build({'type': constants.KEY_CORNERS_2D_STABLE})
        reg_targets_batch = coder.encode_batch(label_boxes_3d, proposals, p2,
                                               image_info)
        reg_targets_batch[match == -1] = 0
        # no need grad_fn
        return reg_targets_batch


@TARGET_ASSIGNERS.register(constants.KEY_CORNERS_3D)
class Corners3DTargetAssigner(RegTargetAssigner):
    @classmethod
    def assign_target(cls, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        # label_boxes_2d = kwargs[constants.KEY_BOXES_2D]
        proposals = kwargs[constants.KEY_PROPOSALS]
        label_boxes_3d = kwargs[constants.KEY_BOXES_3D]
        p2 = kwargs[constants.KEY_STEREO_CALIB_P2]

        # prepare coder
        # 2d coder config
        coder = bbox_coders.build({'type': constants.KEY_CORNERS_3D})
        label_boxes_3d = cls.generate_assigned_label(
            cls, kwargs[constants.KEY_MATCH], label_boxes_3d)
        reg_targets_batch = coder.encode_batch(label_boxes_3d, proposals, p2)

        reg_targets_batch[match == -1] = 0
        # no need grad_fn
        return reg_targets_batch


@TARGET_ASSIGNERS.register(constants.KEY_CORNERS_3D_GRNET)
class Corners3DGRNetTargetAssigner(RegTargetAssigner):
    @classmethod
    def assign_target(cls, **kwargs):
        # match = kwargs[constants.KEY_MATCH]
        label_boxes_3d = kwargs[constants.KEY_BOXES_3D]
        p2 = kwargs[constants.KEY_STEREO_CALIB_P2]

        # prepare coder
        # 2d coder config
        coder = bbox_coders.build({'type': constants.KEY_CORNERS_3D_GRNET})

        reg_targets_batch = coder.encode_batch(label_boxes_3d, p2)
        reg_targets_batch = cls.generate_assigned_label(
            cls, kwargs[constants.KEY_MATCH], reg_targets_batch)

        # reg_targets_batch[match == -1] = 0
        # no need grad_fn
        return reg_targets_batch


@TARGET_ASSIGNERS.register(constants.KEY_MOBILEYE)
class MobilEyeTargetAssigner(RegTargetAssigner):
    @classmethod
    def assign_target(cls, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        label_boxes_3d = kwargs[constants.KEY_BOXES_3D]
        p2 = kwargs[constants.KEY_STEREO_CALIB_P2]
        image_info = kwargs[constants.KEY_IMAGE_INFO]
        proposals = kwargs[constants.KEY_PROPOSALS]
        label_boxes_2d = kwargs[constants.KEY_BOXES_2D]

        # prepare coder
        # 2d coder config
        coder = bbox_coders.build({'type': constants.KEY_MOBILEYE})
        label_boxes_3d = cls.generate_assigned_label(
            cls, kwargs[constants.KEY_MATCH], label_boxes_3d)
        label_boxes_2d = cls.generate_assigned_label(
            cls, kwargs[constants.KEY_MATCH], label_boxes_2d)

        reg_targets_batch = coder.encode_batch(label_boxes_3d, proposals, p2,
                                               image_info, label_boxes_2d)

        reg_targets_batch[match == -1] = 0
        # no need grad_fn
        return reg_targets_batch


@TARGET_ASSIGNERS.register(constants.KEY_MONO_3D_NON_2D_PROJ)
class FPNMono3DNon2DProjTargetAssigner(RegTargetAssigner):
    @classmethod
    def assign_target(cls, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        label_boxes_2d = kwargs[constants.KEY_BOXES_2D]
        label_boxes_3d = kwargs[constants.KEY_BOXES_3D]
        p2 = kwargs[constants.KEY_STEREO_CALIB_P2]

        # prepare coder
        # 2d coder config
        coder = bbox_coders.build({'type': constants.KEY_MONO_3D_NON_2D_PROJ})
        reg_targets_batch = coder.encode_batch(label_boxes_3d, label_boxes_2d,
                                               p2)

        reg_targets_batch = cls.generate_assigned_label(
            cls, kwargs[constants.KEY_MATCH], reg_targets_batch)
        reg_targets_batch[match == -1] = 0
        # no need grad_fn
        return reg_targets_batch


@TARGET_ASSIGNERS.register(constants.KEY_KEYPOINTS)
class KeyPointTargetAssigner(RegTargetAssigner):
    @classmethod
    def assign_target(cls, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        keypoints = kwargs[constants.KEY_KEYPOINTS]

        # prepare coder
        # 2d coder config
        coder = bbox_coders.build({'type': constants.KEY_KEYPOINTS_HEATMAP})
        proposals = kwargs[constants.KEY_PROPOSALS]

        # assign label keypoints first
        assigned_keypoints = cls.generate_assigned_label(
            cls, kwargs[constants.KEY_MATCH], keypoints)
        reg_targets_batch = coder.encode_batch(proposals, assigned_keypoints)

        # reg_targets_batch = cls.generate_assigned_label(
        # cls, kwargs[constants.KEY_MATCH], reg_targets_batch)
        reg_targets_batch[match == -1] = 0
        # no need grad_fn
        return reg_targets_batch
