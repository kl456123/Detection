# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
from abc import ABCMeta, abstractmethod

IMG_EXTENSIONS = [
    '.jpg',
    '.JPG',
    '.jpeg',
    '.JPEG',
    '.png',
    '.PNG',
    '.ppm',
    '.PPM',
    '.bmp',
    '.BMP',
]


class DetDataset(Dataset):
    """
    The important thing is that data and label should be separated
    so that it can adapt to both training and testing mode
    """
    __metaclass__ = ABCMeta

    def __init__(self, training):
        self.imgs = None
        self.scale = None
        self.is_gray = None
        self.data_path = None
        self.transforms = None
        self.num_classes = None
        self.training = training

        # preprocess
        self.preprocess_load_data()
        if not training:
            self.preprocess_load_label()

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item_idx):
        """
        return data and label directly
        """
        data = self.provide_data(item_idx)
        if not self.training:
            label = self.provide_label(item_idx)
        return data, label

    @staticmethod
    def is_image_file(filename):
        return any(
            filename.endswith(extension) for extension in IMG_EXTENSIONS)

    @abstractmethod
    def provide_data(self, item_idx):
        """
        defined by user
        """
        pass

    @abstractmethod
    def provide_label(self, item_idx):
        """
        defined by user
        """
        pass

    def preprocess_load_label(self):
        """
        It can be overload
        """
        pass

    def preprocess_load_data(self):
        """
        It can be overload
        """
        pass
