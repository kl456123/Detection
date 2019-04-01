# -*- coding: utf-8 -*-

import torch
from core import constants


class Analyzer(object):
    @staticmethod
    def analyze_recall(match, num_instances):
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
            gt_mask[match_per_img[match_per_img > -1]] = 1
            num_matched = num_matched + gt_mask.sum()

        stats = {}
        stats[constants.KEY_STATS_NUM_INSTANCES] = (num_matched, num_gt)
        return stats

    @staticmethod
    def analyze_precision(match, rcnn_cls_probs, num_gt, thresh=0.5):
        # import ipdb
        # ipdb.set_trace()
        if match.dim() == 2:
            assert match.shape[0] == 1
            match = match[0]
        num_det = torch.nonzero(rcnn_cls_probs > thresh).numel()
        num_tp = torch.nonzero((rcnn_cls_probs > thresh) & (match > -1
                                                            )).numel()

        match_thresh = match[(rcnn_cls_probs > thresh) & (match > -1)]

        gt_mask = torch.zeros(num_gt).type_as(match_thresh)
        gt_mask[match_thresh] = 1
        matched_thresh = gt_mask.sum().item()
        recall_thresh = matched_thresh / num_gt
        # if num_det:
        # precision = num_tp/num_det
        # else:
        # precision = 0
        return {
            'num_tp': num_tp,
            'num_det': num_det,

            # recall after thresh
            'matched_thresh': matched_thresh,
            'recall_thresh': recall_thresh
        }
