# -*- coding: utf-8 -*-
"""
Combine samples and target_assigner
"""
from abc import ABC, abstractmethod
from core.utils.analyzer import Analyzer

import samplers
import target_assigners
from core import constants
from utils import batch_ops


class TargetGenerator(ABC):
    def __init__(self, target_generator_config):
        self.target_assigner = target_assigners.build(
            target_generator_config['target_assigner_config'])
        self.sampler = samplers.build(
            target_generator_config['sampler_config'])
        self.analyzer = Analyzer(target_generator_config['analyzer_config'])

    def generate_targets(self, proposals_dict, gt_dict, num_instances):
        """
            use gt to encode preds for better predictable
        Args:
            proposals: shape(N, M, 4)
            feed_dict: dict

        Returns:
            targets_list: list, [(target1, weight1),(target2, weight2)]
        """
        ##########################
        # matcher
        ##########################

        ##########################
        # assigner
        ##########################

        loss_units = self.target_assigner.assign(proposals_dict, gt_dict, num_instances)

        ##########################
        # subsampler
        ##########################
        cls_criterion = None
        reg_weights = loss_units[constants.KEY_BOXES_2D]['weight']
        cls_weights = loss_units[constants.KEY_CLASSES]['weight']

        pos_indicator = reg_weights > 0
        indicator = cls_weights > 0

        # subsample from all
        # shape (N,M)
        batch_sampled_mask = self.sampler.subsample_batch(
            pos_indicator, indicator=indicator, criterion=cls_criterion)

        # dict
        proposals_dict = batch_ops.filter_tensor_container(proposals_dict,
                                                           batch_sampled_mask)

        # list
        loss_units = batch_ops.filter_tensor_container(loss_units,
                                                       batch_sampled_mask)
        # add pred for loss_unit
        for key in proposals_dict:
            if key == constants.KEY_PRIMARY:
                continue
            loss_units[key]['pred'] = proposals_dict[key]

        return proposals_dict, loss_units
