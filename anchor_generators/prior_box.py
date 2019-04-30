# Descartes, basic object detection laboratory
# Support python2.7, python3, based on Pytorch 1.0
# Author: Yang Maoke (maokeyang@deepmotion.ai)
# Copyright (c) 2019-present, DeepMotion


import math
import torch
import numpy
import itertools


class RetinaPriorBox(object):
    """
        * Compute priorbox coordinates in center-offset form for each source feature map.
    """
    def __init__(self):
        super(RetinaPriorBox, self).__init__()

    def __call__(self, cfg):
        self.image_size = cfg['input_shape']
        self.aspect_ratios = cfg['aspect_ratio']
        self.default_ratio = cfg['default_ratio']
        self.output_stride = cfg['output_scale']
        self.clip = True

        scale_w = self.image_size[0]
        scale_h = self.image_size[1]
        img_ratio = scale_w / scale_h
        steps_w = [s / scale_w for s in self.output_stride]
        steps_h = [s / scale_h for s in self.output_stride]
        sizes = self.default_ratio
        aspect_ratios = self.aspect_ratios

        feature_map_w = [int(math.floor(scale_w / s)) for s in self.output_stride]
        feature_map_h = [int(math.floor(scale_h / s)) for s in self.output_stride]
        assert len(feature_map_h) == len(feature_map_w)
        num_layers = len(feature_map_h)

        boxes = []
        for i in range(num_layers):
            fm_w = feature_map_w[i]
            fm_h = feature_map_h[i]
            for h, w in itertools.product(range(fm_h), range(fm_w)):
                cx = (w + 0.5) * steps_w[i]
                cy = (h + 0.5) * steps_h[i]

                s = sizes[i]
                boxes.append((cx, cy, s, s))

                for ar in aspect_ratios[i]:
                    boxes.append((cx, cy, s * math.sqrt(ar), img_ratio * s / math.sqrt(ar)))
                    boxes.append((cx, cy, s / math.sqrt(ar), img_ratio * s * math.sqrt(ar)))

                s = sizes[i] * math.pow(2, 1./3)
                boxes.append((cx, cy, s, s))

                for ar in aspect_ratios[i]:
                    boxes.append((cx, cy, s * math.sqrt(ar), img_ratio * s / math.sqrt(ar)))
                    boxes.append((cx, cy, s / math.sqrt(ar), img_ratio * s * math.sqrt(ar)))

                s = sizes[i] * math.pow(2, 2./3)
                boxes.append((cx, cy, s, s))

                for ar in aspect_ratios[i]:
                    boxes.append((cx, cy, s * math.sqrt(ar), img_ratio * s / math.sqrt(ar)))
                    boxes.append((cx, cy, s / math.sqrt(ar), img_ratio * s * math.sqrt(ar)))

        boxes = numpy.array(boxes, dtype=float)
        boxes = torch.from_numpy(boxes).float()  # back to torch land
        if self.clip:
            boxes.clamp_(min=0., max=1.)
        return boxes


class NanoRetinaPriorBox(object):
    """
        * Compute priorbox coordinates in center-offset form for each source feature map.
    """
    def __init__(self):
        super(NanoRetinaPriorBox, self).__init__()

    def __call__(self, cfg):
        self.image_size = cfg['input_shape']
        self.aspect_ratios = cfg['aspect_ratio']
        self.default_ratio = cfg['default_ratio']
        self.output_stride = cfg['output_scale']
        self.clip = True

        scale_w = self.image_size[0]
        scale_h = self.image_size[1]
        img_ratio = scale_w / scale_h
        steps_w = [s / scale_w for s in self.output_stride]
        steps_h = [s / scale_h for s in self.output_stride]
        sizes = self.default_ratio
        aspect_ratios = self.aspect_ratios

        feature_map_w = [int(math.floor(scale_w / s)) for s in self.output_stride]
        feature_map_h = [int(math.floor(scale_h / s)) for s in self.output_stride]
        assert len(feature_map_h) == len(feature_map_w)
        num_layers = len(feature_map_h)

        boxes = []
        for i in range(num_layers):
            fm_w = feature_map_w[i]
            fm_h = feature_map_h[i]
            for h, w in itertools.product(range(fm_h), range(fm_w)):
                cx = (w + 0.5) * steps_w[i]
                cy = (h + 0.5) * steps_h[i]

                s = sizes[i]
                for ar in aspect_ratios[i]:
                    boxes.append((cx, cy, s * math.sqrt(ar), img_ratio * s / math.sqrt(ar)))
                    boxes.append((cx, cy, s / math.sqrt(ar), img_ratio * s * math.sqrt(ar)))

                s = sizes[i] * math.pow(2, 1./3)
                for ar in aspect_ratios[i]:
                    boxes.append((cx, cy, s * math.sqrt(ar), img_ratio * s / math.sqrt(ar)))
                    boxes.append((cx, cy, s / math.sqrt(ar), img_ratio * s * math.sqrt(ar)))

                s = sizes[i] * math.pow(2, 2./3)
                for ar in aspect_ratios[i]:
                    boxes.append((cx, cy, s * math.sqrt(ar), img_ratio * s / math.sqrt(ar)))
                    boxes.append((cx, cy, s / math.sqrt(ar), img_ratio * s * math.sqrt(ar)))

        boxes = numpy.array(boxes, dtype=float)
        boxes = torch.from_numpy(boxes).float()  # back to torch land
        if self.clip:
            boxes.clamp_(min=0., max=1.)
        return boxes
