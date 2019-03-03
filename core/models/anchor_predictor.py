# -*- coding: utf-8 -*-

from core.model import Model
import torch.nn as nn


class AnchorPredictor(Model):
    def init_param(self, model_config):
        self.keep_prob = model_config['keep_prob']
        self.pooling_size = model_config['pooling_size']

    def init_modules(self):
        self.cls_fc6 = nn.Conv2d(1, 256, 3, 1, 0)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout2d(self.keep_prob)
        self.cls_fc7 = nn.Conv2d(256, 256, 1, 1, 0)
        self.relu7 = nn.ReLU()
        self.dropout7 = nn.Dropout2d(self.keep_prob)
        self.cls_fc8 = nn.Conv2d(256, 2, 1, 1, 0)
        self.relu8 = nn.ReLU()

        self.reg_fc6 = nn.Conv2d(1, 256, 3, 1, 0)
        self.relu9 = nn.ReLU()
        self.dropout9 = nn.Dropout2d(self.keep_prob)
        self.reg_fc7 = nn.Conv2d(256, 256, 1, 1, 0)
        self.relu10 = nn.ReLU()
        self.dropout10 = nn.Dropout2d(self.keep_prob)
        self.reg_fc8 = nn.Conv2d(256, 6, 1, 1, 0)
        self.relu11 = nn.ReLU()

    def forward(self, x):
        # cls
        net = self.cls_fc6(x)
        net = self.relu6(net)
        net = self.dropout6(net)

        net = self.cls_fc7(net)
        net = self.relu7(net)
        net = self.dropout7(net)

        objectness = self.cls_fc8(net)
        objectness = objectness.squeeze()

        # reg
        net = self.reg_fc6(x)
        net = self.relu9(net)
        net = self.dropout9(net)

        net = self.reg_fc7(net)
        net = self.relu10(net)
        net = self.dropout10(net)

        offset = self.reg_fc8(net)
        offset = offset.squeeze()
        return objectness, offset
