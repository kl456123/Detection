# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod


class DetDataset(Dataset, metaclass=ABCMeta):
    """
    The important thing is that data and label should be separated
    so that it can adapt to both training and testing mode
    """

    def __init__(self, training):
        self.sample_names = None
        self.transforms = None
        self.training = training

    def __len__(self):
        return len(self.sample_names)

    @abstractmethod
    def __getitem__(self, item_idx):
        """
        return data and label directly
        """
        pass

    @abstractmethod
    def get_sample(self, idx):
        pass
