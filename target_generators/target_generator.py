# -*- coding: utf-8 -*-
"""
Combine samples and target_assigner
"""
from abc import ABC, abstractmethod
from core.utils.analyzer import Analyzer

import samplers
import target_assigners
from core import constants
import torch
from utils import batch_ops


class TargetGenerator(ABC):
    def __init__(self, target_generator_config):
        self.target_assigner = target_assigners.build(
            target_generator_config['target_assigner_config'])
        self.sampler = samplers.build(
            target_generator_config['sampler_config'])
        self.analyzer = Analyzer(target_generator_config['analyzer_config'])

    def generate_targets(self, proposals_dict, gt_dict):
        """
            use gt to encode preds for better predictable
        Args:
            proposals: shape(N, M, 4)
            feed_dict: dict

        Returns:
            targets_list: list, [(target1, weight1),(target2, weight2)]
        """
        batch_size = gt_dict[constants.KEY_PRIMARY].shape[0]

        ##########################
        # assigner
        ##########################
        loss_units = self.target_assigner.assign(proposals_dict, gt_dict)

        ##########################
        # subsampler
        ##########################
        cls_criterion = None
        reg_weights = loss_units[1]['weight']
        cls_weights = loss_units[0]['weight']
        cls_target = loss_units[0]['target']
        reg_target = loss_units[1]['target']

        pos_indicator = reg_weights > 0
        indicator = cls_weights > 0

        # subsample from all
        # shape (N,M)
        batch_sampled_mask = self.sampler.subsample_batch(
            pos_indicator, indicator=indicator, criterion=cls_criterion)

        # dict
        proposals_dict = batch_ops.filter_tensor_container(proposals_dict)
        # list
        loss_units = batch_ops.filter_tensor_container(loss_units)

        return proposals_dict, loss_units
