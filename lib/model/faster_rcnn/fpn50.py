# -*- coding: utf-8 -*-

import torch.nn as nn

from models.fpn import fpn50
# add libs to sys path
from model.faster_rcnn.faster_rcnn import _fasterRCNN


class FPN(_fasterRCNN):
    def __init__(self, model_config):
        pass

    def _init_modules(self):
        fpn = fpn50()
        # output: p2,p3,p4,p5
        self.RCNN_base = fpn

        self.RCNN_cls_score = nn.Linear(2048, self.n_classes)
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(2048, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(2048, 4 * self.n_classes)

    def _init_weights(self):
        pass
