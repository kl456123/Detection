# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod
import logging
import os
import numpy as np


class DetDataset(Dataset, metaclass=ABCMeta):
    """
    The important thing is that data and label should be separated
    so that it can adapt to both training and testing mode
    """

    def __init__(self, training, logger=None):
        self.sample_names = None
        self.transforms = None
        self.training = training
        if logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, index):
        sample = self.get_sample(index)
        if self.transforms is not None:
            sample = self.transforms(sample)

        if self.training:
            sample = self.pad_sample(sample)

        return sample

    def pad_sample(self, sample):
        return sample

    @abstractmethod
    def get_training_sample(self, idx):
        pass

    @abstractmethod
    def get_testing_sample(self, idx):
        pass

    def get_sample(self, idx):
        if self.training:
            sample = self.get_training_sample(idx)
        else:
            sample = self.get_testing_sample(idx)
        return sample

    @staticmethod
    def load_sample_names_from_image_dir(image_dir):
        images_path = []
        for img_name in sorted(os.listdir(image_dir)):
            images_path.append(os.path.join(image_dir, img_name))
        return np.array(images_path)

    @staticmethod
    def load_sample_names_from_dataset_file(image_dir, dataset_file):
        # set_file = './train.txt'
        with open(dataset_file) as f:
            sample_names = f.read().splitlines()

        sample_names = [
            os.path.join(image_dir, "{}.png".format(sample_name))
            for sample_name in sample_names
        ]
        return np.array(sample_names)

    @staticmethod
    def get_sample_name_from_path(sample_path):
        return os.path.splitext(os.path.basename(sample_path))[0]
