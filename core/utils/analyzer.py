# -*- coding: utf-8 -*-

import torch
from core import constants


class Analyzer(object):
    @staticmethod
    def analyze_recall(match, num_instances, append_num=10):
        """
        Args:
            match: shape(N, M)
            num_instances: shape(N, )
        Returns:
            stats: dict
        """
        batch_size = match.shape[0]
        num_matched = 0
        num_gt = num_instances.sum()
        for batch_ind in range(batch_size):
            match_per_img = match[batch_ind]
            num_instances_per_img = num_instances[batch_ind]
            gt_mask = torch.zeros(num_instances_per_img.item()).type_as(match)
            # remove appended gt first
            match_per_img = match_per_img[:-append_num]
            gt_mask[match_per_img[match_per_img > -1]] = 1
            num_matched = num_matched + gt_mask.sum()

        stats = {}
        stats[constants.KEY_STATS_RECALL] = torch.tensor(
            [num_matched, num_gt]).to('cuda').float().unsqueeze(0)
        return stats

    @staticmethod
    def analyze_precision(match, rcnn_cls_probs, num_instances, thresh=0.5):
        batch_size = match.shape[0]
        num_dets = 0
        num_gt = num_instances.sum()
        num_matched_thresh = 0
        num_tps = 0
        for batch_ind in range(batch_size):
            cls_probs_per_img = rcnn_cls_probs[batch_ind]
            match_per_img = match[batch_ind]
            num_det = torch.nonzero(cls_probs_per_img > thresh).numel()
            num_tp = torch.nonzero((cls_probs_per_img > thresh) & (
                match_per_img > -1)).numel()
            num_tps = num_tp + num_tps
            match_thresh = match[(cls_probs_per_img > thresh) & (match_per_img
                                                                 > -1)]

            gt_mask = torch.zeros(
                num_instances[batch_ind]).type_as(match_thresh)
            gt_mask[match_thresh] = 1
            num_matched_thresh = num_matched_thresh + gt_mask.sum()

            num_dets = num_det + num_dets

        stats = {}
        stats[constants.KEY_STATS_PRECISION] = torch.tensor(
            num_tps, num_dets).to('cuda').float().unsqueeze(0)
        stats[constants.KEY_STATS_THRESH_RECALL] = torch(
            num_matched_thresh, num_gt).to('cuda').float().unsqueeze(0)
        return stats
