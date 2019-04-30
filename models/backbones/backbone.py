# Descartes, basic object detection laboratory
# Support python2.7, python3, based on Pytorch 1.0
# Author: Yang Maoke (maokeyang@deepmotion.ai)
# Copyright (c) 2019-present, DeepMotion


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import BatchNorm2d as bn


class AggregationBlock(nn.Module):

    def __init__(self, n_channels1, n_channels2, n_channels3, det_features=256):
        """
        :param n_channels1:
        :param n_channels2:
        :param n_channels3:
        """
        super(AggregationBlock, self).__init__()
        self.squeeze1 = nn.Conv2d(in_channels=n_channels1, out_channels=det_features, kernel_size=1)
        self.squeeze2 = nn.Conv2d(in_channels=n_channels2, out_channels=det_features, kernel_size=1)
        self.squeeze3 = nn.Conv2d(in_channels=n_channels3, out_channels=det_features, kernel_size=1)

        self.node1 = nn.Sequential(
            nn.Conv2d(in_channels=det_features, out_channels=det_features, kernel_size=3, padding=1),
            bn(num_features=det_features),
            nn.ReLU(inplace=True)
        )

        self.node2 = nn.Sequential(
            nn.Conv2d(in_channels=det_features, out_channels=det_features, kernel_size=3, padding=1),
            bn(num_features=det_features),
            nn.ReLU(inplace=True)
        )

        self.node3 = nn.Sequential(
            nn.Conv2d(in_channels=det_features, out_channels=det_features, kernel_size=3, padding=1),
            bn(num_features=det_features),
            nn.ReLU(inplace=True)
        )

        self.fusing = nn.Sequential(
            nn.Conv2d(in_channels=det_features * 3, out_channels=det_features, kernel_size=1),
        )

    def forward(self, c3, c4, c5):
        c3 = self.squeeze1(c3)
        c4 = self.squeeze2(c4)
        c5 = self.squeeze3(c5)

        c3 = self.node1.forward(c3 + F.interpolate(c4, scale_factor=2, mode='bilinear'))
        c4 = self.node2.forward(c4 + F.interpolate(c5, scale_factor=2, mode='bilinear'))
        c3 = self.node3.forward(c3 + F.interpolate(c4, scale_factor=2, mode='bilinear'))

        c3 = torch.cat([c3,
                        F.interpolate(c4, scale_factor=2, mode='bilinear'),
                        F.interpolate(c5, scale_factor=4, mode='bilinear')],
                       dim=1)

        c3 = self.fusing(c3)

        return c3


class AffinityPropagate(nn.Module):

    def __init__(self, inters=5):
        super(AffinityPropagate, self).__init__()
        self.inters = inters

    def forward(self, guidance, feature_maps):
        # normalize features
        gate1 = F.softmax(guidance.narrow(1, 0, 5), dim=1)
        gate2 = F.softmax(guidance.narrow(1, 5, 5), dim=1)
        gate3 = F.softmax(guidance.narrow(1, 10, 5), dim=1)
        gate4 = F.softmax(guidance.narrow(1, 15, 5), dim=1)
        gate5 = F.softmax(guidance.narrow(1, 20, 5), dim=1)

        for i in range(self.inters):
            # one propagation
            feature_map_0 = self.propagation(gate1, feature_maps)
            feature_map_1 = self.propagation(gate2, feature_maps)
            feature_map_2 = self.propagation(gate3, feature_maps)
            feature_map_3 = self.propagation(gate4, feature_maps)
            feature_map_4 = self.propagation(gate5, feature_maps)

            feature_maps[0] = feature_map_0 + feature_maps[0]
            feature_maps[1] = feature_map_1 + feature_maps[1]
            feature_maps[2] = feature_map_2 + feature_maps[2]
            feature_maps[3] = feature_map_3 + feature_maps[3]
            feature_maps[4] = feature_map_4 + feature_maps[4]

        return feature_maps

    def propagation(self, gate, feature_map):
        gate1 = gate.narrow(1, 0, 1)
        feature = gate1 * feature_map[0]

        for i in range(4):
            idx = i + 1
            gate1 = gate.narrow(1, idx, 1)
            feature = feature + gate1 * feature_map[idx]

        return feature

    @staticmethod
    def normalize(gate):
        """
        normalize to (-1, 1)
        :param gate:
        :return:
        """
        mean = torch.mean(gate, dim=1).unsqueeze(1)
        max_gate, _ = torch.max(gate, dim=1)
        min_gate, _ = torch.min(gate, dim=1)
        max_gate = max_gate.unsqueeze(1)
        min_gate = min_gate.unsqueeze(1)

        gate = (gate - mean) / (max_gate - min_gate + 0.0000001)
        return gate

    @staticmethod
    def wightning(gate):
        mean = torch.mean(gate, dim=1).unsqueeze(1)
        diff = gate - mean
        bias = torch.sqrt(torch.mean(torch.pow(diff, 2), dim=1).unsqueeze(1))
        return diff / bias

