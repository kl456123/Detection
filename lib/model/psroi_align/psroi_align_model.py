# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
from .._ext import psroi_pooling


class PSRoIAlignFunction(nn.F):
    @staticmethod
    def forward(features, rois, output_dim, pooled_width, pooled_height,
                spatial_scale, group_size):
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]
        output = torch.zeros(num_rois, output_dim, pooled_height, pooled_width)
        mappingchannel = torch.IntTensor(num_rois, output_dim, pooled_height,
                                         pooled_width).zero_()
        output = output.cuda()
        mappingchannel = mappingchannel.cuda()
        psroi_pooling.psroi_pooling_forward_cuda(
            pooled_height, pooled_width, spatial_scale, group_size, output_dim,
            features, rois, output, mappingchannel)

        return output

    @staticmethod
    def backward(self):
        pass


class PSRoIAlignModel(nn.Module):
    def __init__(self, pooled_width, pooled_height, spatial_scale, group_size,
                 output_dim):
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.group_size = int(group_size)
        self.output_dim = int(output_dim)

        self.mappingchannel = None
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        PSRoIAlignFunction.apply(features, rois)
