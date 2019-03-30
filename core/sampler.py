# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import torch
from .utils import format_checker


class Sampler(ABC):
    def __init__(self, sampler_config):
        """
        Note that the scores here is the critation of samples,it can be
        confidence or IoU e,c
        """
        self.fg_fraction = sampler_config['fg_fraction']
        self.num_samples = sampler_config['num_samples']

    @abstractmethod
    def subsample(self,
                  num_samples,
                  pos_indicator,
                  criterion=None,
                  indicator=None):
        pass

    def subsample_batch(self,
                        pos_indicator,
                        criterion=None,
                        indicator=None):
        """
            batch version of subsample
        """
        pos_indicator = pos_indicator.detach()
        if indicator is None:
            indicator = torch.ones_like(pos_indicator)
        indicator = indicator.detach()

        # check format
        format_checker.check_tensor_dims(pos_indicator, 2)
        format_checker.check_tensor_dims(indicator, 2)

        batch_size = pos_indicator.shape[0]
        if criterion is None:
            criterion = [None] * batch_size
        else:
            criterion = criterion.detach()

        num_samples_per_img = self.num_samples // batch_size

        sample_mask = []
        for i in range(batch_size):
            sample_mask.append(
                self.subsample(
                    num_samples_per_img,
                    pos_indicator[i],
                    criterion=criterion[i],
                    indicator=indicator[i]))

        sample_mask = torch.stack(sample_mask)
        return sample_mask
