# -*- coding: utf-8 -*-

import torch


class Analyzer(object):
    def __init__(self):
        pass

    def analyze(self, match):
        """
        analyze result from match,calculate AP and AR
        note that -1 means it is not matched
        Args:
            match: tensor(N,M)

        """
        num_all_samples = match.numel()
        num_unmatched_samples = torch.nonzero(
            match[match == -1]).view(-1).numel()
        num_matched_samples = num_all_samples - num_unmatched_samples
        # print('match rate: ', num_matched_samples / num_all_samples)
