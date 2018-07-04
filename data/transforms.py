import torch
import random
import numpy as np

import matplotlib
from PIL import Image, ImageFilter


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) * (box_b[3]-box_b[1]))              # [A,B]
    union = area_a + area_b - inter
    return inter / union                                              # [A,B]


class RandomHSV(object):
    """
    Args:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
    Returns:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self, h_range=(1.0, 1.0), s_range=(0.7, 1.3), v_range=(0.7, 1.3), ratio=0.5):
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range
        self.ratio = ratio

    def __call__(self, img, boxes=None, labels=None):
        rand_value = random.randint(1, 100)
        if rand_value > 100 * self.ratio:
            return img, boxes, labels

        img = np.array(img)
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h*h_random, 0, 1)
        img_s = np.clip(img_s*s_random, 0, 1)
        img_v = np.clip(img_v*v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)

        return img_new, boxes, labels


class RandomSampleCrop(object):
    """
    Args:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Returns:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self, min_aspect, max_aspect, keep_aspect=True):
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.keep_aspect = keep_aspect
        self.sample_options = (
            # Using entire original input image.
            None,
            # Sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9.
            # (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            # Randomly sample a patch.
            # (None, None),
            'zoom_out',
        )

    def __call__(self, image, boxes=None, labels=None):
        width, height, = image.size
        while True:
            # Randomly choose a mode.
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            if mode is 'zoom_out':
                # place the image on a 1.5X mean pic
                # 0.485, 0.456, 0.406
                mean_img = np.zeros((int(1.5*height), int(1.5*width), 3))
                mean_img[:, :, 0] = np.uint8(0.485 * 255)
                mean_img[:, :, 1] = np.uint8(0.456 * 255)
                mean_img[:, :, 2] = np.uint8(0.406 * 255)

                left = np.random.uniform(0, 0.5) * width
                top = np.random.uniform(0, 0.5) * height
                rect = np.array([int(left), int(top), int(left+width), int(top+height)])
                mean_img[rect[1]:rect[3], rect[0]:rect[2], :] = np.array(image)
                # mask = boxes[:, 3] > (boxes[:, 1] + 30)
                # current_boxes = boxes[mask, :].copy()  # take only matching gt boxes
                # current_labels = labels[mask].copy()

                current_boxes = boxes.copy()
                current_labels = labels.copy()
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] += rect[:2]
                current_boxes[:, 2:] += rect[:2]

                return Image.fromarray(mean_img.astype(np.uint8)), current_boxes, current_labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # Max trails (50), or change mode.
            for _ in range(50):
                current_image = np.array(image)

                if self.keep_aspect:
                    current_image = np.array(image)
                    w = np.random.uniform(0.8 * width, width)
                    h = w * height / float(width)

                    # Convert to integer rect x1,y1,x2,y2.
                    left = np.random.uniform(width - w)
                    top = left / width * height
                else:
                    w = np.random.uniform(0.7 * width, width)
                    h = np.random.uniform(0.7 * height, height)
                    # Aspect ratio constraint b/t .5 & 2.
                    if h / w < self.min_aspect or h / w > self.max_aspect:
                        continue

                    # Convert to integer rect x1,y1,x2,y2.
                    left = np.random.uniform(width - w)
                    top = np.random.uniform(height - h)

                rect = np.array([int(left), int(top), int(left + w), int(top + h)])
                # Calculate IoU (jaccard overlap) b/t the cropped and gt boxes.
                overlap = jaccard_numpy(boxes, rect)
                # Is min and max overlap constraint satisfied? if not try again.
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # Cut the crop from the image.
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]
                # Keep overlap with gt box IF center in sampled patch.
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                # Mask in all gt boxes that above and to the left of centers.
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                # Mask in all gt boxes that under and to the right of centers.
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                # mask in that both m1 and m2 are true
                mask = m1 * m2
                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                current_boxes = boxes[mask, :].copy()  # take only matching gt boxes
                current_labels = labels[mask]          # take only matching gt labels

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return Image.fromarray(current_image), current_boxes, current_labels


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

    def __call__(self, tensor, bbox, label):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor, bbox, label


class RandomBrightness(object):
    def __init__(self, shift_value=30):
        self.shift_value = shift_value

    def __call__(self, img, bbox, label):
        shift = np.random.uniform(-self.shift_value, self.shift_value, size=1)
        image = np.array(img, dtype=float)
        image[:, :, :] += shift
        image = np.around(image)
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        return image, bbox, label


class RandomGaussBlur(object):
    def __init__(self, max_blur=4):
        self.max_blur = max_blur

    def __call__(self, img, bbox, label):
        blur_value = np.random.uniform(0, self.max_blur)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_value))
        return img, bbox, label


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img, bbox, label):
        """
        Args:
            img (PIL.Image): Image to be flipped.
            bbox[np.array]: bbox to be flipped

        Returns:
            PIL.Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            w, h = img.size
            xmin = w - bbox[:, 2]
            xmax = w - bbox[:, 0]
            bbox[:, 0] = xmin
            bbox[:, 2] = xmax
            return img.transpose(Image.FLIP_LEFT_RIGHT), bbox, label
        else:
            return img, bbox, label


class Resize(object):
    """random Rescale the input PIL.Image to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, _size, interpolation=Image.BICUBIC):
        # assert isinstance(_size, int)
        self.new_size = _size
        self.interpolation = interpolation

    def __call__(self, img, bbox, label):
        """
        Args:
            img (PIL.Image): Image to be scaled.

        Returns:
            PIL.Image: Rescaled image.
        """
        w, h = img.size
        bbox[:, 2] /= w
        bbox[:, 0] /= w
        bbox[:, 1] /= h
        bbox[:, 3] /= h

        return img.resize((self.new_size[0], self.new_size[1]), self.interpolation), bbox, label


class ToTensor(object):
    """Convert a ``PIL.Image`` to tensor.
    Converts a PIL.Image in the range [0, 255] to a torch.FloatTensor
    of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __call__(self, pic, bbox, label):
        """
        Args:
            pic (PIL.Image): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        pic = np.array(pic)
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        lbl = torch.from_numpy(label).long()
        bbox = torch.from_numpy(bbox).float()

        return img.float().div(255), bbox, lbl


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bbox, label):
        for t in self.transforms:
            img, bbox, label = t(img, bbox, label)
        return img, bbox, label
