# Descartes, basic object detection laboratory
# Support python2.7, python3, based on Pytorch 1.0
# Author: Yang Maoke (maokeyang@deepmotion.ai)
# Copyright (c) 2019-present, DeepMotion

import torch
from torch import nn
from torch.nn import BatchNorm2d as bn


class TwoStageRetinaLayer(nn.Module):
    def __init__(self, num_classes, num_anchor, num_regress=4,
                 in_channels=256):
        super(TwoStageRetinaLayer, self).__init__()
        self.num_anchors = num_anchor
        self.num_regress = num_regress
        self.num_classes = num_classes

        self.loc_feature1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.loc_feature2 = nn.Sequential(
            nn.Conv2d(
                in_channels * 2, in_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.cls_feature1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.cls_feature2 = nn.Sequential(
            nn.Conv2d(
                in_channels * 2, in_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.os_out = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_anchors * 2,
            kernel_size=1)
        self.cls_out = nn.Conv2d(
            in_channels, self.num_anchors * self.num_classes, kernel_size=1)
        self.box_out1 = nn.Conv2d(
            in_channels,
            out_channels=self.num_anchors * self.num_regress,
            kernel_size=1)
        self.box_out2 = nn.Conv2d(
            in_channels,
            out_channels=self.num_anchors * self.num_regress,
            kernel_size=1)

    def forward(self, features):
        y_locs1 = []
        y_locs2 = []
        y_os = []
        y_cls = []

        for i, x in enumerate(features):
            # location out
            loc_feature = self.loc_feature1(x)
            loc1 = self.box_out1(loc_feature)

            N = loc1.size(0)
            loc1 = loc1.permute(0, 2, 3, 1).contiguous()
            loc1 = loc1.view(N, -1, self.num_regress)
            y_locs1.append(loc1)

            loc_feature = torch.cat([x, loc_feature], dim=1)
            loc_feature = self.loc_feature2(loc_feature)
            loc2 = self.box_out2(loc_feature)

            N = loc2.size(0)
            loc2 = loc2.permute(0, 2, 3, 1).contiguous()
            loc2 = loc2.view(N, -1, self.num_regress)
            loc2 += loc1
            y_locs2.append(loc2)

            # os out
            cls_feature = self.cls_feature1(x)
            os_out = self.os_out(cls_feature)
            os_out = os_out.permute(0, 2, 3, 1).contiguous()
            # _size = os_out.size(1)
            os_out = os_out.view(N, -1, 2)
            y_os.append(os_out)

            cls_feature = torch.cat([x, cls_feature], dim=1)
            cls_feature = self.cls_feature2(cls_feature)
            cls_out = self.cls_out(cls_feature)

            cls_out = cls_out.permute(0, 2, 3, 1).contiguous()
            cls_out = cls_out.view(N, -1, self.num_classes)
            y_cls.append(cls_out)

        loc1_preds = torch.cat(y_locs1, dim=1)
        loc2_preds = torch.cat(y_locs2, dim=1)
        os_preds = torch.cat(y_os, dim=1)
        cls_preds = torch.cat(y_cls, dim=1)

        return loc1_preds, loc2_preds, os_preds, cls_preds


class StridePoolBlock(nn.Sequential):
    def __init__(self, input_num, out_num, stride=2):
        super(StridePoolBlock, self).__init__()
        self.add_module(
            'conv3',
            nn.Conv2d(
                in_channels=input_num,
                out_channels=out_num,
                kernel_size=3,
                stride=stride,
                padding=1))
        self.add_module('bn', bn(num_features=out_num))
        self.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, _input):
        _feature = super(StridePoolBlock, self).forward(_input)

        return _feature
