# -*- coding: utf-8 -*-
from torch.utils.data import Dataset

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
    def __init__(self):
        self.imgs = None
        self.scale = None
        self.is_gray = None
        self.data_path = None
        self.transforms = None
        self.num_classes = None

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item_idx):
        pass

    @staticmethod
    def is_image_file(filename):
        return any(
            filename.endswith(extension) for extension in IMG_EXTENSIONS)
