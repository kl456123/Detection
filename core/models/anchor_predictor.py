# -*- coding: utf-8 -*-

from core.model import Model
import torch.nn as nn
from core.filler import Filler


class AnchorPredictor(Model):
    def init_param(self, model_config):
        self.keep_prob = model_config['keep_prob']
        self.pooling_size = model_config['pooling_size']
        self.ndin = model_config['ndin']
        self.norm_std = 0.0001
        self.norm_mean = 0
        self.truncated = True

    def init_weights(self):
        pass
        # for layer in [
                # self.cls_fc6, self.cls_fc7, self.cls_fc8, self.reg_fc6,
                # self.reg_fc7, self.reg_fc8
        # ]:
            # Filler.normal_init(layer, self.norm_mean, self.norm_std,
                               # self.truncated)

    def init_modules(self):
        self.cls_fc6 = nn.Conv2d(self.ndin, 256, 3, 1, 0)
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout2d(self.keep_prob)
        self.cls_fc7 = nn.Conv2d(256, 256, 1, 1, 0)
        self.bn7 = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU()
        self.dropout7 = nn.Dropout2d(self.keep_prob)
        self.cls_fc8 = nn.Conv2d(256, 2, 1, 1, 0)

        self.reg_fc6 = nn.Conv2d(self.ndin, 256, 3, 1, 0)
        self.bn9 = nn.BatchNorm2d(256)
        self.relu9 = nn.ReLU()
        self.dropout9 = nn.Dropout2d(self.keep_prob)
        self.reg_fc7 = nn.Conv2d(256, 256, 1, 1, 0)
        self.bn10 = nn.BatchNorm2d(256)
        self.relu10 = nn.ReLU()
        self.dropout10 = nn.Dropout2d(self.keep_prob)
        self.reg_fc8 = nn.Conv2d(256, 6, 1, 1, 0)

    def forward(self, x):
        # cls
        net = self.cls_fc6(x)
        net = self.bn6(net)
        net = self.relu6(net)
        #  net = self.dropout6(net)

        net = self.cls_fc7(net)
        net = self.bn7(net)
        net = self.relu7(net)
        #  net = self.dropout7(net)

        objectness = self.cls_fc8(net)
        objectness = objectness.view(-1, 2)

        # reg
        net = self.reg_fc6(x)
        net = self.bn9(net)
        net = self.relu9(net)
        #  net = self.dropout9(net)

        net = self.reg_fc7(net)
        net = self.bn10(net)
        net = self.relu10(net)
        #  net = self.dropout10(net)

        offset = self.reg_fc8(net)
        offset = offset.view(-1, 6)
        return objectness, offset
