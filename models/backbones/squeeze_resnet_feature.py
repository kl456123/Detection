# Descartes, basic object detection laboratory
# Support python2.7, python3, based on Pytorch 1.0
# Author: Yang Maoke (maokeyang@deepmotion.ai)
# Copyright (c) 2019-present, DeepMotion


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.backbone import AggregationBlock

from torch.nn import BatchNorm2d as bn


class SqueezeResNetFeature(nn.Module):

    # Add pruning_rate in function __init__
    def __init__(self, block, layers, pruning_rate=0.5, det_features=128, dla_input=None):
        self.inplanes = 64
        super(SqueezeResNetFeature, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.layer1 = self._make_layer(block, 64, layers[0], pruning_rate, stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], pruning_rate, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], pruning_rate, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], pruning_rate, stride=2)

        self.aggregation = AggregationBlock(n_channels1=dla_input[0], n_channels2=dla_input[1],
                                            n_channels3=dla_input[2], det_features=det_features)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, bn):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, pruning_rate, stride=1, dilation=1, padding=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                bn(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, pruning_rate, stride, dilation, padding, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, pruning_rate, dilation=dilation, padding=padding))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)  # [H/2, W/2, 64]
        x = self.maxpool(x)                                # [H/4, W/4, 64]
        x = self.layer1(x)                                 # [H/4, W/4, 64]

        layer2 = self.layer2(x)                            # [H/8, W/8, 128]
        layer3 = self.layer3(layer2)                       # [H/16, W/16, 256]
        layer4 = self.layer4(layer3)                       # [H/32, W/32, 512]

        c3 = self.aggregation.forward(c3=layer2, c4=layer3, c5=layer4)

        return c3

    def load_pretrained_weight(self, net):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in net.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


class BasicBlock(nn.Module):
    expansion = 1

    # Add pruning_rate in function BasicBlock()
    def __init__(self, inplanes, planes, pruning_rate, stride=1, dilation=1, padding=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.pruned_channel_planes = int(planes - math.floor(planes * pruning_rate))
        self.conv1 = conv3x3(inplanes, self.pruned_channel_planes, stride, dilation, padding)
        self.bn1 = bn(self.pruned_channel_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(self.pruned_channel_planes, planes)
        self.bn2 = bn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     dilation=dilation, padding=padding, bias=False)


if __name__ == '__main__':
    net = SqueezeResNetFeature(BasicBlock, [2, 2, 2, 2], 0.5)
    weights = torch.load("../pretrained/resnet18_pruned0.5.pth")
    net.load_pretrained_weight(weights)
    x = torch.randn(1, 3, 384, 768).cuda()
    net.eval().cuda()
    c3 = net.forward(x)
    print(c3.size())
