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
        #  import ipdb
        #  ipdb.set_trace()
        targets = self.target_assigner.assign(proposals_dict, gt_dict)

        ##########################
        # subsampler
        ##########################
        cls_criterion = None
        reg_weights = targets[1]['weight']
        cls_weights = targets[0]['weight']
        cls_target = targets[0]['target']
        reg_target = targets[1]['target']

        pos_indicator = reg_weights > 0
        indicator = cls_weights > 0

        # subsample from all
        # shape (N,M)
        batch_sampled_mask = self.sampler.subsample_batch(
            pos_indicator, indicator=indicator, criterion=cls_criterion)

        cls_weights = cls_weights[batch_sampled_mask].view(batch_size, -1)
        reg_weights = reg_weights[batch_sampled_mask].view(batch_size, -1)
        num_cls_coeff = (cls_weights > 0).sum(dim=-1)
        num_reg_coeff = (reg_weights > 0).sum(dim=-1)

        if num_reg_coeff == 0:
            num_reg_coeff = torch.ones_like(num_reg_coeff)
        if num_cls_coeff == 0:
            num_cls_coeff = torch.ones_like(num_cls_coeff)

        # targets = [target.update() for target in targets]
        targets[0]['weight'] = cls_weights / num_cls_coeff.float()
        targets[1]['weight'] = reg_weights / num_reg_coeff.float()

        targets[0]['target'] = cls_target[batch_sampled_mask].view(
            batch_size, -1)
        targets[1]['target'] = reg_target[batch_sampled_mask].view(
            batch_size, -1, reg_target.shape[-1])

        proposals = proposals_dict[constants.KEY_PRIMARY]
        proposals = proposals[batch_sampled_mask].view(batch_size, -1, 4)

        return proposals, targets, batch_sampled_mask
