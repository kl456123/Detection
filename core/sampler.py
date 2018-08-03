# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod


class Sampler(object):
    __meta__ = ABC

    def __init__(self, sampler_config):
        """
        Note that the scores here is the critation of samples,it can be
        confidence or IoU e,c
        """
        self.fg_fraction = sampler_config['fg_fraction']
        self.num_samples = sampler_config['num_samples']
        self.thresh = sampler_config['thresh']

    def calculate_num(self):
        # if num_samples is not None:
        num_samples = self.num_samples
        # self.num_samples = num_samples
        self.fg_num = self.fg_fraction * num_samples
        self.bg_num = num_samples - self.fg_num
        self.num_samples = num_samples

    def subsample(self, mask, overlaps, scores=None):
        # two samples methods
        if scores is not None:
            critation = scores
        else:
            critation = overlaps
        pos_mask = mask and (overlaps > self.thresh)
        self.calculate_num()
        self._subsample(mask, pos_mask, critation)

    @abstractmethod
    def _subsample(self, mask, pos_mask, scores):
        """
        Returns two mask, one is named mb_mask used for masking mb and non-mb
        the other is named mb_pos_mask used for masking pos and neg in mb_mask

        Reurns:
        mb_mask,
        mb_pos_mask,

        """
        num_samples = self.num_samples
        self.calculate_num(num_samples)

        mb_mask = None
        mb_pos_mask = None

        return mb_mask, mb_pos_mask
