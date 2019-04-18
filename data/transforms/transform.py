import torch
import random
import numpy as np
import matplotlib
from PIL import Image, ImageFilter
from torchvision.transforms import functional as F

from core import constants
from core.utils import format_checker
from utils.registry import TRANSFORMS
from abc import ABC, abstractmethod
from utils import geometry_utils
from utils.geometry_utils import ProjectMatrixTransform


class Transform(object):
    def __init__(self, config):
        pass

    @abstractmethod
    def __call__(self, sample):
        raise NotImplementedError


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
    area_a = (
        (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


@TRANSFORMS.register('to_pil')
class ToPILImage(Transform):
    """Convert a tensor or an ndarray to PIL Image.

    Converts a torch.*Tensor of shape C x H x W or a numpy ndarray of shape
    H x W x C to a PIL Image while preserving the value range.

    Args:
    mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).
        If ``mode`` is ``None`` (default) there are some assumptions made about the input data:
        1. If the input has 3 channels, the ``mode`` is assumed to be ``RGB``.
        2. If the input has 4 channels, the ``mode`` is assumed to be ``RGBA``.
        3. If the input has 1 channel, the ``mode`` is determined by the data type (i,e,
        ``int``, ``float``, ``short``).

    .. _PIL.Image mode: http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#modes
    """

    def __call__(self, sample):
        """
        Args:
            pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.

        Returns:
            PIL Image: Image converted to PIL Image.

        """
        pic = sample[constants.KEY_IMAGE]
        sample[constants.KEY_IMAGE] = F.to_pil_image(pic, mode='RGB')
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        if self.mode is not None:
            format_string += 'mode={0}'.format(self.mode)
        format_string += ')'
        return format_string


@TRANSFORMS.register('random_hsv')
class RandomHSV(Transform):
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

    def __init__(self, config):
        h_range = config.get('h_range', (1.0, 1.0))
        s_range = config.get('s_range', (0.7, 1.3))
        v_range = config.get('v_range', (0.7, 1.3))
        ratio = config.get('ratio', 0.5)
        self.h_range = h_range
        self.s_range = s_range
        self.v_range = v_range
        self.ratio = ratio

    def __call__(self, sample):
        img = sample[constants.KEY_IMAGE]
        format_checker.check_pil_image(img)
        rand_value = random.randint(1, 100)
        if rand_value > 100 * self.ratio:
            return sample

        img = np.array(img)
        img_hsv = matplotlib.colors.rgb_to_hsv(img)
        img_h, img_s, img_v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :,
                                                                          2]
        h_random = np.random.uniform(min(self.h_range), max(self.h_range))
        s_random = np.random.uniform(min(self.s_range), max(self.s_range))
        v_random = np.random.uniform(min(self.v_range), max(self.v_range))
        img_h = np.clip(img_h * h_random, 0, 1)
        img_s = np.clip(img_s * s_random, 0, 1)
        img_v = np.clip(img_v * v_random, 0, 255)
        img_hsv = np.stack([img_h, img_s, img_v], axis=2)
        img_new = matplotlib.colors.hsv_to_rgb(img_hsv)
        sample[constants.KEY_IMAGE] = Image.fromarray(img_new.astype(np.uint8))

        return sample


@TRANSFORMS.register('random_zoomout')
class RandomZoomOut(Transform):
    def __init__(self, config):
        self.scale = config.get('scale', 1.5)

    def __call__(self, sample):
        image = sample[constants.KEY_IMAGE]
        format_checker.check_pil_image(image)

        width, height, = image.size
        boxes = sample[constants.KEY_LABEL_BOXES_2D]
        labels = sample[constants.KEY_LABEL_CLASSES]
        # place the image on a 1.5X mean pic
        # 0.485, 0.456, 0.406
        scale = self.scale
        remain = scale - 1
        assert remain > 0, 'scale must be greater than 1.0'
        mean_img = np.zeros((int(scale * height), int(scale * width), 3))
        mean_img[:, :, 0] = np.uint8(0.485 * 255)
        mean_img[:, :, 1] = np.uint8(0.456 * 255)
        mean_img[:, :, 2] = np.uint8(0.406 * 255)
        left = np.random.uniform(0, remain) * width
        top = np.random.uniform(0, remain) * height
        rect = np.array(
            [int(left), int(top), int(left + width), int(top + height)])
        mean_img[rect[1]:rect[3], rect[0]:rect[2], :] = np.array(image)

        current_boxes = boxes.copy()
        current_labels = labels.copy()
        # adjust to crop (by substracting crop's left,top)
        current_boxes[:, :2] += rect[:2]
        current_boxes[:, 2:] += rect[:2]
        sample[constants.KEY_IMAGE] = Image.fromarray(
            mean_img.astype(np.uint8))
        sample[constants.KEY_LABEL_BOXES_2D] = current_boxes
        sample[constants.KEY_LABEL_CLASSES] = current_labels
        return sample


@TRANSFORMS.register('random_sample_crop')
class RandomSampleCrop(Transform):
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

    def __init__(self, config):
        self.min_aspect = config.get('min_aspect')
        self.max_aspect = config.get('max_aspect')
        self.keep_aspect = config.get('keep_aspect', True)
        self.sample_options = (
            # Using entire original input image.
            None,
            # Sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9.
            # (0.1, None),
            #  (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None))

    def __call__(self, sample):
        raise NotImplementedError('there are some bugs here')
        image = sample[constants.KEY_IMAGE]
        format_checker.check_pil_image(image)

        width, height, = image.size
        boxes = sample[constants.KEY_LABEL_BOXES_2D]
        labels = sample[constants.KEY_LABEL_CLASSES]
        while True:
            # Randomly choose a mode.
            mode = random.choice(self.sample_options)
            if mode is None:
                return sample

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

                rect = np.array(
                    [int(left), int(top), int(left + w), int(top + h)])
                # Calculate IoU (jaccard overlap) b/t the cropped and gt boxes.
                overlap = jaccard_numpy(boxes, rect)
                # Is min and max overlap constraint satisfied? if not try again.
                # if overlap.min() < min_iou and max_iou < overlap.max():
                    # continue

                # Cut the crop from the image.
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[
                    2], :]
                # Keep overlap with gt box IF center in sampled patch.
                # centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                # # Mask in all gt boxes that above and to the left of centers.
                # m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                # # Mask in all gt boxes that under and to the right of centers.
                # m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                # mask in that both m1 and m2 are true
                # mask = m1 * m2
                # mask = np.ones_like(mask, dtype=mask.dtype)
                mask = overlap > 0
                # import ipdb
                # ipdb.set_trace()
                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                current_boxes = boxes[mask, :].copy(
                )  # take only matching gt boxes
                current_labels = labels[mask]  # take only matching gt labels

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]
                # print('before croped shape: ',image.size)
                # print('after croped shape: ',current_image.shape)

                sample[constants.KEY_IMAGE] = Image.fromarray(
                    current_image.astype(np.uint8))
                sample[constants.KEY_LABEL_BOXES_2D] = current_boxes
                sample[constants.KEY_LABEL_CLASSES] = current_labels
                if sample.get(constants.KEY_LABEL_BOXES_3D) is not None:
                    label_boxes_3d = sample[constants.KEY_LABEL_BOXES_3D]
                    sample[constants.KEY_LABEL_BOXES_3D] = label_boxes_3d[mask]

                if sample.get(constants.KEY_STEREO_CALIB_P2) is not None:
                    p2 = sample[constants.KEY_STEREO_CALIB_P2]
                    new_p2 = ProjectMatrixTransform.crop(p2, rect[:2])

                    sample[constants.KEY_STEREO_CALIB_P2] = new_p2
                return sample


@TRANSFORMS.register('normalize')
class Normalize(Transform):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
    mean (sequence): Sequence of means for R, G, B channels respecitvely.
    std (sequence): Sequence of standard deviations for R, G, B channels.
    """

    def __init__(self, config):
        self.mean = config['normal_mean']
        self.std = config['normal_std']

    def __call__(self, sample):
        """
        Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
        Tensor: Normalized image.
        """
        tensor = sample[constants.KEY_IMAGE]
        format_checker.check_tensor(tensor)
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        sample[constants.KEY_IMAGE] = tensor
        return sample


@TRANSFORMS.register('random_brightness')
class RandomBrightness(Transform):
    def __init__(self, config):
        self.shift_value = config.get('shift_value', 30)

    def __call__(self, sample):
        img = sample[constants.KEY_IMAGE]
        format_checker.check_pil_image(img)
        shift = np.random.uniform(-self.shift_value, self.shift_value, size=1)
        image = np.array(img, dtype=float)
        image[:, :, :] += shift
        image = np.around(image)
        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)
        image = Image.fromarray(image)
        sample[constants.KEY_IMAGE] = image
        return sample


@TRANSFORMS.register('random_gaussian_blur')
class RandomGaussBlur(Transform):
    def __init__(self, config):
        self.max_blur = config.get('max_blur', 4)

    def __call__(self, sample):
        img = sample[constants.KEY_IMAGE]
        format_checker.check_pil_image(img)
        blur_value = np.random.uniform(0, self.max_blur)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_value))
        sample[constants.KEY_IMAGE] = img
        return sample


@TRANSFORMS.register('random_horizontal_flip')
class RandomHorizontalFlip(Transform):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        """
        Args:
        img (PIL.Image): Image to be flipped.
        bbox[np.array]: bbox to be flipped

        Returns:
        PIL.Image: Randomly flipped image.
        """
        img = sample[constants.KEY_IMAGE]
        format_checker.check_pil_image(img)
        bbox = sample[constants.KEY_LABEL_BOXES_2D]
        if random.random() < 0.5:
            w, h = img.size
            xmin = w - bbox[:, 2]
            xmax = w - bbox[:, 0]
            bbox[:, 0] = xmin
            bbox[:, 2] = xmax
            sample[constants.KEY_IMAGE] = img.transpose(Image.FLIP_LEFT_RIGHT)
            sample[constants.KEY_LABEL_BOXES_2D] = bbox

            if sample.get(constants.KEY_STEREO_CALIB_P2) is not None:
                p2 = sample[constants.KEY_STEREO_CALIB_P2]
                w = img.size[0]
                new_p2 = ProjectMatrixTransform.horizontal_flip(p2, w)
                sample[constants.KEY_STEREO_CALIB_P2] = new_p2

        return sample


@TRANSFORMS.register('fix_ratio_resize')
class FixRatioResize(object):
    def __init__(self, config):
        pass

    def __call__(self, sample):
        pass


@TRANSFORMS.register('fix_shape_resize')
class FixShapeResize(object):
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
    interpolate_map = {'bilinear': Image.BILINEAR, 'bicubic': Image.BICUBIC}

    def __init__(self, config):
        interpolate = config.get('interpolate', 'bilinear')
        self.interpolation = self.interpolate_map[interpolate]
        # h, w
        self.size = config['size']

    def __call__(self, sample):
        """
        Args:
        img (PIL.Image): Image to be scaled.
        Returns:
        PIL.Image: Rescaled image.
        """
        img = sample[constants.KEY_IMAGE]
        format_checker.check_pil_image(img)
        w, h = img.size
        # w, h
        target_shape = (int(self.size[1]), int(self.size[0]))
        # h, w
        image_scale = [self.size[0] / h, self.size[1] / w]

        # resized image
        sample[constants.KEY_IMAGE] = img.resize(target_shape,
                                                 self.interpolation)

        # image_info
        image_info = sample[constants.KEY_IMAGE_INFO]
        image_info[:2] = self.size
        image_info[2:4] = image_scale
        sample[constants.KEY_IMAGE_INFO] = image_info

        # bbox
        if sample.get(constants.KEY_LABEL_BOXES_2D) is not None:
            bbox = sample[constants.KEY_LABEL_BOXES_2D]
            bbox[:, ::2] *= image_scale[1]
            bbox[:, 1::2] *= image_scale[0]

            sample[constants.KEY_LABEL_BOXES_2D] = bbox

        if sample.get(constants.KEY_STEREO_CALIB_P2) is not None:
            # import ipdb
            # ipdb.set_trace()
            p2 = sample[constants.KEY_STEREO_CALIB_P2]
            # K, T = geometry_utils.decompose_matrix(p2)
            # K[0, :] = K[0, :] * image_scale[1]
            # K[1, :] = K[1, :] * image_scale[0]
            # K[2, 2] = 1
            # KT = np.dot(K, T)

            # new_p2 = np.zeros((3, 4)).astype(p2.dtype)
            # # assign back
            # new_p2[:3, :3] = K
            # new_p2[:, 3] = KT
            new_p2 = ProjectMatrixTransform.resize(p2, image_scale)
            sample[constants.KEY_STEREO_CALIB_P2] = new_p2

        return sample


@TRANSFORMS.register('resize')
class Resize(Transform):
    def __init__(self, config):
        min_size = config.get('min_size')
        max_size = config.get('max_size')
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size, )

        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(
                    round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, sample):
        import ipdb
        ipdb.set_trace()

        image = sample[constants.KEY_IMAGE]
        format_checker.check_pil_image(image)
        target = sample[constants.KEY_LABEL_BOXES_2D]
        size = self.get_size(image.size)
        image = F.resize(image, size)
        target = target.resize(image.size)

        image_info = sample[constants.KEY_IMAGE_INFO]
        image_info[:2] = size
        w, h = image.size
        image_scale = (size[0] / h, size[1] / w)
        image_info[2:4] = image_scale
        sample[constants.KEY_IMAGE_INFO] = image_info

        sample[constants.KEY_IMAGE] = image
        sample[constants.KEY_LABEL_BOXES_2D] = target
        return sample


@TRANSFORMS.register('to_tensor')
class ToTensor(Transform):
    """Convert a ``PIL.Image`` to tensor.
    Converts a PIL.Image in the range [0, 255] to a torch.FloatTensor
    of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, sample):
        """
        Args:
        pic (PIL.Image): Image to be converted to tensor.
        Returns:
        Tensor: Converted image.
        """
        image = sample[constants.KEY_IMAGE]
        format_checker.check_pil_image(image)
        image = F.to_tensor(image)
        sample[constants.KEY_IMAGE] = image

        return sample
