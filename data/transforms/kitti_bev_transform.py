import torch
import random
import numpy as np


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img, bbox, ry, label):
        """
        Args:
            img[np.array]:
            bbox[np.array]: bbox to be flipped

        Returns:
            PIL.Image: Randomly flipped image.
        """
        h, w = img.shape[:2]
        if random.random() < 0.5:
            xmin = w - bbox[:, 2]
            xmax = w - bbox[:, 0]
            bbox[:, 0] = xmin
            bbox[:, 2] = xmax

            ry[ry >=0] = np.pi - ry[ry >= 0]
            ry[ry < 0] = -np.pi - ry[ry < 0]

            return np.flip(img, axis=1).copy(), bbox, ry, label
        else:
            return img, bbox, ry, label

class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, bbox, ry, label):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor, bbox, ry, label

class ToTensor(object):
    """Convert a ``PIL.Image`` to tensor.
    Converts a PIL.Image in the range [0, 255] to a torch.FloatTensor
    of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __call__(self, img, bbox, ry, label):
        """
        Args:
            img(numpy.array):
        Returns:
            Tensor: Converted image.
        """
        h, w = img.shape[:2]
        bbox[:, 2] /= w
        bbox[:, 0] /= w
        bbox[:, 1] /= h
        bbox[:, 3] /= h
        img = torch.from_numpy(img)
        img = img.permute(2, 1, 0)
        lbl = torch.from_numpy(label).long()
        bbox = torch.from_numpy(bbox).float()

        return img.float(), bbox, ry, lbl

class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bbox, ry, label):
        for t in self.transforms:
            img, bbox, ry, label = t(img, bbox, ry, label)
        return img, bbox, ry, label
