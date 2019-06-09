# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import torch

import bbox_coders
from core import constants
from utils.registry import TARGET_ASSIGNERS


class TargetAssigner(ABC):
    def __init__(self, config):
        self.coder = bbox_coders.build(config)

    def assign_targets_and_weights(self, **kwargs):
        targets, inside_weights = self._assign_targets_and_inside_weights(
            **kwargs)
        match = kwargs[constants.KEY_MATCH]
        outside_weight = self.assign_outside_weight(match)
        # float type
        if inside_weights is not None:
            weights = outside_weight * inside_weights
        return targets, weights

    @abstractmethod
    def _assign_targets_and_inside_weights(self, **kwargs):
        """
        if dont change weights just return None as the second item
        """
        pass

    def _assign_outside_weight(self, match):
        """
        Args:
            match: shape(N, M), -1 refers to no anyone matched
        """
        return torch.ones_like(match).float()


class RegTargetAssigner(TargetAssigner):
    def _assign_outside_weight(self, match):
        reg_weights = super().assign_weight(match)
        reg_weights[match == -1] = 0
        return reg_weights


@TARGET_ASSIGNERS.register(constants.KEY_CLASSES)
class ClassesTargetAssigner(TargetAssigner):
    def __init__(self, config):
        self.coder = bbox_coders.build(config)
        self.bg_thresh = config.get('bg_thresh', 0)

    def _assign_targets_and_inside_weights(self, **kwargs):
        match = kwargs[constants.KEY_MATCH]
        gt = torch.ones_like(kwargs[constants.KEY_CLASSES])
        assigned_gt = self.generate_assigned_label(kwargs[constants.KEY_MATCH],
                                                   gt)
        assigned_gt[match == -1] = 0

        assigned_overlaps_batch = kwargs[constants.KEY_ASSIGNED_OVERLAPS]
        bg_thresh = self.bg_thresh
        # assign inside weights
        cls_weights = torch.ones_like(assigned_overlaps_batch)
        if bg_thresh > 0:
            ignored_bg = (assigned_overlaps_batch > bg_thresh) & (match == -1)
            cls_weights[ignored_bg] = 0

        return assigned_gt.long(), cls_weights


@TARGET_ASSIGNERS.register(constants.KEY_OBJECTNESS)
class ObjectnessTargetAssigner(ClassesTargetAssigner):
    def _assign_targets_and_inside_weights(self, **kwargs):
        targets, weights = super()._assign_targets_and_inside_weights(**kwargs)
        targets[targets > 1] = 1
        return targets, weights


class InstanceAssigner(dict):
    def __init__(self, config):
        super().__init__()
        for attr_name in config:
            self[attr_name] = TARGET_ASSIGNERS[attr_name](config[attr_name])


class LossDict(dict):
    KEY_PREDS = 'preds'
    KEY_TARGETS = 'targets'
    KEY_WEIGHTS = 'weights'

    def update_loss_unit(self, name, loss_unit):
        if self.get(name) is not None:
            self[name].update(loss_unit)
        else:
            self[name] = dict()

    def update_from_output(self, output_dict):
        for key in self:
            if key in output_dict:
                preds = output_dict[key]
                self.update_loss_unit(key, {'preds': preds})

    def get_preds(self, attr_name):
        return self[attr_name][self.KEY_PREDS]

    def get_targets(self, attr_name):
        return self[attr_name][self.KEY_TARGETS]

    def get_weights(self, attr_name):
        return self[attr_name][self.KEY_WEIGHTS]


class Instance(object):
    def __init__(self, config):
        self._instance_assigner = InstanceAssigner(config[config['assigner']])
        self._instance_losses = Loss(config['loss'])
        # self.losses = {}

    def generate_losses(self, output_dict, feed_dict, auxiliary_dict):
        proposals_primary = output_dict[constants.KEY_PRIMARY].detach()
        gt_primary = feed_dict[constants.KEY_PRIMARY].detach()

        # match them
        match_quality_matrix = self.similarity_calc.compare_batch(
            proposals_primary, gt_primary)
        num_instances = auxiliary_dict[constants.KEY_NUM_INSTANCES]
        match, assigned_overlaps_batch = self.matcher.match_batch(
            match_quality_matrix, num_instances, self.fg_thresh)

        losses = LossDict()
        # assign targets and weights
        for attr_name in self._instance_assigner:
            assigner = self._instance_assigner[attr_name]
            # generate preds, targets and weights
            targets, weights = assigner.assign_targets_and_weights(
                feed_dict, auxiliary_dict)
            preds = output_dict.get(attr_name)

            # update losses
            losses.update_loss_unit(
                attr_name, {
                    LossDict.TARGETS: targets,
                    LossDict.WEIGHTS: weights,
                    LossDict.PREDS: preds
                })

        return losses

    def calc_loss(self, losses):
        loss_dict = dict()
        for attr_name in self._instance_losses:
            instance_loss_fn = self._instance_losses[attr_name]
            loss_dict[attr_name] = instance_loss_fn[losses[attr_name]]
        return loss_dict
