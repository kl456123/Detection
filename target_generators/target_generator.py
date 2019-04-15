# -*- coding: utf-8 -*-
"""
Combine samples and target_assigner
"""
from core.utils.analyzer import Analyzer

import samplers
import coders
from core import constants
from utils import batch_ops
import matchers
import similarity_calcs
import torch


class TargetGenerator(object):
    def __init__(self, target_generator_config):
        # self.target_assigner = coders.build(
        # target_generator_config['target_assigner_config'])
        self.sampler = samplers.build(
            target_generator_config['sampler_config'])
        self.similarity_calc = similarity_calcs.build(
            target_generator_config['similarity_calc_config'])
        self.matcher = matchers.build(
            target_generator_config['matcher_config'])

        self.bg_thresh = target_generator_config['bg_thresh']
        self.fg_thresh = target_generator_config['fg_thresh']

        self.stats = {}

    def suppress_ignored_case(self, match, num_instances):
        """
        Args:
            match: shape(N, M)
            num_instances: shape(N, ), it determines the num of valid instances,
            it refers to the last dim of match_quality_matrix
        """
        m = match.clone()
        m[match == -1] = 0
        return m

    def generate_targets(self,
                         proposals_dict,
                         gt_dict,
                         auxiliary_dict,
                         device='cuda',
                         subsample=True):
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
        # usually IoU overlaps is used as metric
        proposals_primary = proposals_dict[constants.KEY_PRIMARY].detach()
        gt_primary = gt_dict[constants.KEY_PRIMARY].detach()

        match_quality_matrix = self.similarity_calc.compare_batch(
            proposals_primary, gt_primary)

        match = self.matcher.match_batch(match_quality_matrix, self.fg_thresh)
        assigned_overlaps_batch = self.matcher.assigned_overlaps_batch.to(
            device)

        # get recall stats
        num_instances = auxiliary_dict[constants.KEY_NUM_INSTANCES]
        fake_match = self.matcher.match_batch(match_quality_matrix, 0.7)
        self.stats.update(Analyzer.analyze_recall(fake_match, num_instances))

        ignored_match = self.suppress_ignored_case(match, num_instances)

        ##########################
        # assigner
        ##########################
        # generate_targets and weight
        loss_units = {}
        for key in gt_dict:
            if key == constants.KEY_PRIMARY:
                continue
            target_assigner_config = {'type': key}
            target_assigner = coders.build(target_assigner_config)
            # some match results used for encoding
            kwargs = {
                constants.KEY_BG_THRESH: self.bg_thresh,
                # no any ignored case will be assigned
                constants.KEY_MATCH: match,
                constants.KEY_IGNORED_MATCH: ignored_match,
                constants.KEY_ASSIGNED_OVERLAPS: assigned_overlaps_batch
            }
            kwargs.update(auxiliary_dict)
            # weight_args = [match, assigned_overlaps_batch]

            weight = target_assigner.assign_weight(**kwargs)
            target = target_assigner.assign_target(**kwargs)
            loss_units[key] = {
                'weight': weight,
                'target': target,
            }

        # loss_units = self.target_assigner.assign(proposals_dict, gt_dict,
        # num_instances)

        ##########################
        # subsampler
        ##########################
        if subsample:
            cls_criterion = None
            reg_weights = loss_units[constants.KEY_BOXES_2D]['weight']
            cls_weights = loss_units[constants.KEY_CLASSES]['weight']

            pos_indicator = reg_weights > 0
            indicator = cls_weights > 0

            # subsample from all
            # shape (N,M)
            assert indicator[indicator].numel(
            ) >= self.sampler.num_samples, 'no enough samples before subsample'
            batch_sampled_mask = self.sampler.subsample_batch(
                pos_indicator, indicator=indicator, criterion=cls_criterion)

            assert batch_sampled_mask[batch_sampled_mask].numel(
            ) == self.sampler.num_samples, 'not enough samples after subsample'

            # dict
            proposals_dict = batch_ops.filter_tensor_container(proposals_dict,
                                                            batch_sampled_mask)

            # list
            loss_units = batch_ops.filter_tensor_container(loss_units,
                                                        batch_sampled_mask)
        # generate pred
        # add pred for loss_unit
        for key in proposals_dict:
            if key == constants.KEY_PRIMARY:
                continue
            loss_units[key]['pred'] = proposals_dict[key]

        return proposals_dict, loss_units
