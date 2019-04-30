# -*- coding: utf-8 -*-

import torch

import itertools
import math
import numpy as np

from utils.registry import ANCHOR_GENERATORS
from utils import geometry_utils


@ANCHOR_GENERATORS.register('retina')
class AnchorGenerator(object):
    def __init__(self, cfg):
        """
        """
        self.aspect_ratios = cfg['aspect_ratio']
        self.default_ratio = cfg['default_ratio']
        self.output_stride = cfg['output_scale']

        self.num_anchors = (1 + 2 * len(self.aspect_ratios[0])) * 3

    def generate(self, input_size, normalize=False):
        scale_h = input_size[0]
        scale_w = input_size[1]
        img_ratio = scale_w / scale_h
        steps_w = [s / scale_w for s in self.output_stride]
        steps_h = [s / scale_h for s in self.output_stride]
        sizes = self.default_ratio
        aspect_ratios = self.aspect_ratios

        feature_map_w = [
            int(math.floor(scale_w / s)) for s in self.output_stride
        ]
        feature_map_h = [
            int(math.floor(scale_h / s)) for s in self.output_stride
        ]
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
                    boxes.append((cx, cy, s * math.sqrt(ar),
                                  img_ratio * s / math.sqrt(ar)))
                    boxes.append((cx, cy, s / math.sqrt(ar),
                                  img_ratio * s * math.sqrt(ar)))

                s = sizes[i] * math.pow(2, 1. / 3)
                boxes.append((cx, cy, s, s))

                for ar in aspect_ratios[i]:
                    boxes.append((cx, cy, s * math.sqrt(ar),
                                  img_ratio * s / math.sqrt(ar)))
                    boxes.append((cx, cy, s / math.sqrt(ar),
                                  img_ratio * s * math.sqrt(ar)))

                s = sizes[i] * math.pow(2, 2. / 3)
                boxes.append((cx, cy, s, s))

                for ar in aspect_ratios[i]:
                    boxes.append((cx, cy, s * math.sqrt(ar),
                                  img_ratio * s / math.sqrt(ar)))
                    boxes.append((cx, cy, s / math.sqrt(ar),
                                  img_ratio * s * math.sqrt(ar)))

        boxes = np.array(boxes, dtype=float)
        boxes = torch.from_numpy(boxes).float()  # back to torch land

        if not normalize:
            # unnorm
            boxes[:, ::2] = boxes[:, ::2] * input_size[1]
            boxes[:, 1::2] = boxes[:, 1::2] * input_size[0]
        # use batch format
        boxes = boxes.unsqueeze(0)

        if not normalize:
            pass
        else:
            boxes.clamp_(min=0., max=1.)

        boxes = geometry_utils.torch_xywh_to_xyxy(boxes)

        return boxes
