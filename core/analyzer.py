# -*- coding: utf-8 -*-

import torch


class Analyzer(object):
    def __init__(self):
        pass

    def analyze(self, match, num_gt):
        """
        analyze result from match,calculate AP and AR
        note that -1 means it is not matched
        Args:
            match: tensor(N,M)

        """
        assert match.shape[0] == 1
        match = match[0]
        match = match[match > -1]
        gt_mask = torch.zeros(num_gt).type_as(match)
        gt_mask[match[:-num_gt]] = 1
        matched = gt_mask.sum().item()
        recall = matched / num_gt
        self.stat = {'matched': matched, 'num_gt': num_gt, 'recall': recall}
        return self.stat
        # print('matched_gt/all_gt/average recall({}/{}/{}): '.format(

    # matched, num_gt, recall))
    # num_all_samples = match.numel()
    # num_unmatched_samples = torch.nonzero(

    # match[match == -1]).view(-1).numel()
    # num_matched_samples = num_all_samples - num_unmatched_samples
    # print('match rate: ', num_matched_samples / num_all_samples)
