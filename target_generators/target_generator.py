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

        self.fake_fg_thresh = target_generator_config.get('fake_fg_thresh',
                                                          0.7)
        self.target_generator_config = target_generator_config

        # self.stats = {}

    #  def suppress_ignored_case(self, match, num_instances):
    #  """
    #  Args:
    #  match: shape(N, M)
    #  num_instances: shape(N, ), it determines the num of valid instances,
    #  it refers to the last dim of match_quality_matrix
    #  """
    #  m = match.clone()
    #  m[match == -1] = 0
    #  return m

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
        stats = {}
        ##########################
        # matcher
        ##########################
        # usually IoU overlaps is used as metric
        proposals_primary = proposals_dict[constants.KEY_PRIMARY].detach()
        gt_primary = gt_dict[constants.KEY_PRIMARY].detach()

        # insanity check
        assert not torch.isnan(proposals_primary).any()

        num_instances = auxiliary_dict[constants.KEY_NUM_INSTANCES]
        match_quality_matrix = self.similarity_calc.compare_batch(
            proposals_primary, gt_primary)

        match, assigned_overlaps_batch = self.matcher.match_batch(
            match_quality_matrix, num_instances, self.fg_thresh)

        # get recall stats
        num_instances = auxiliary_dict[constants.KEY_NUM_INSTANCES]
        fake_match, _ = self.matcher.match_batch(
            match_quality_matrix, num_instances, self.fake_fg_thresh)
        # remove appended gts
        append_num_gt = gt_primary.shape[-2]
        stats.update(
            Analyzer.analyze_recall(fake_match, num_instances, append_num_gt))
        auxiliary_dict[constants.KEY_FAKE_MATCH] = fake_match

        # ignored_match = self.suppress_ignored_case(match, num_instances)

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
                # some params used for coders
                constants.KEY_TARGET_GENERATOR_CONFIG:
                self.target_generator_config,
                constants.KEY_BG_THRESH: self.bg_thresh,
                # no any ignored case will be assigned
                constants.KEY_MATCH: match,
                # constants.KEY_IGNORED_MATCH: ignored_match,
                constants.KEY_ASSIGNED_OVERLAPS: assigned_overlaps_batch
            }
            kwargs.update(auxiliary_dict)
            # weight_args = [match, assigned_overlaps_batch]

            target = target_assigner.assign_target(**kwargs)
            weight = target_assigner.assign_weight(**kwargs)
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
            proposals_dict = batch_ops.filter_tensor_container(
                proposals_dict, batch_sampled_mask)

            # list
            loss_units = batch_ops.filter_tensor_container(loss_units,
                                                           batch_sampled_mask)
        # generate pred
        # add pred for loss_unit
        for key in proposals_dict:
            # if key == constants.KEY_PRIMARY:
            # continue
            if key in loss_units:
                # maybe key is not in loss_units
                loss_units[key]['pred'] = proposals_dict[key]

        return proposals_dict, loss_units, stats
