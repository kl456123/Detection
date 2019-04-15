# -*- coding: utf-8 -*-

import torch

import core.ops as ops

from utils.registry import ANCHOR_GENERATORS


@ANCHOR_GENERATORS.register('default')
class AnchorGenerator(object):
    def __init__(self, anchor_generator_config):
        """
        """
        self.base_anchor_size = torch.tensor(
            anchor_generator_config['base_anchor_size'], dtype=torch.float32)
        self.scales = torch.tensor(
            anchor_generator_config['scales'], dtype=torch.float32)
        self.aspect_ratios = torch.tensor(
            anchor_generator_config['aspect_ratios'], dtype=torch.float32)
        self.anchor_offset = torch.tensor(
            anchor_generator_config['anchor_offset'], dtype=torch.float32)

        if anchor_generator_config.get('use_pyramid'):
            self.num_anchors = self.aspect_ratios.numel()
        else:
            self.num_anchors = self.aspect_ratios.numel() * self.scales.numel()

        # depercated,it can make a bug easily
        # self.input_size = anchor_generator_config['input_size']

    def generate_pyramid(self, feature_map_list, input_size, device='cuda'):
        # import ipdb
        # ipdb.set_trace()
        anchors_list = []
        scales = self.scales.view(len(self.scales), -1).expand(
            (-1, len(self.aspect_ratios))).to(device)

        for stage, feature_map_shape in enumerate(feature_map_list):
            anchors_list.append(
                self._generate(feature_map_shape, scales[
                    stage], self.aspect_ratios.to(device), input_size, device))

        return torch.cat(anchors_list, dim=0)

    def generate(self, feature_map_list, input_size, device='cuda'):
        """
        Args:
            feature_map_list, list of (stride, ratio)
        Returns:
            anchors
        """
        anchors_list = []
        scales, aspect_ratios = ops.meshgrid(
            self.scales.to(device), self.aspect_ratios.to(device))
        for feature_map_shape in feature_map_list:
            anchors_list.append(
                self._generate(feature_map_shape, scales, aspect_ratios,
                               input_size, device))

        return torch.cat(anchors_list, dim=0)

    def _generate(self,
                  feature_map_shape,
                  scales,
                  aspect_ratios,
                  input_size,
                  device='cuda'):
        """
        """
        # shape(A,)

        ratios_sqrt = torch.sqrt(aspect_ratios)
        heights = scales * ratios_sqrt * self.base_anchor_size.to(device)
        widths = scales / ratios_sqrt * self.base_anchor_size.to(device)
        anchor_stride = [
            input_size[0] / feature_map_shape[0],
            input_size[1] / feature_map_shape[1]
        ]

        y_ctrs = torch.arange(
            feature_map_shape[0],
            device=device) * anchor_stride[0] + self.anchor_offset[0].to(
                device).float()
        x_ctrs = torch.arange(
            feature_map_shape[1],
            device=device) * anchor_stride[1] + self.anchor_offset[1].to(
                device)
        y_ctrs = y_ctrs.float()
        x_ctrs = x_ctrs.float()

        # meshgrid
        # shape(H*W,)
        x_ctrs, y_ctrs = ops.meshgrid(x_ctrs, y_ctrs)

        # shape(K*A,)
        heights, x_ctrs = ops.meshgrid(heights, x_ctrs)
        widths, y_ctrs = ops.meshgrid(widths, y_ctrs)

        xmin = x_ctrs - 0.5 * (widths - 1)
        ymin = y_ctrs - 0.5 * (heights - 1)
        xmax = x_ctrs + 0.5 * (widths - 1)
        ymax = y_ctrs + 0.5 * (heights - 1)

        # (x,y,h,w)
        return torch.stack([xmin, ymin, xmax, ymax], dim=1)
