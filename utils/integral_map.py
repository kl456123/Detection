# -*- coding: utf-8 -*-

import torch


class IntegralMapGenerator(object):
    @staticmethod
    def generate(input_map):
        """
        Args:
            input_map: shape(NCHW)
        """
        integral_map = input_map.clone()

        integral_map = integral_map.cumsum(dim=-1)
        integral_map = integral_map.cumsum(dim=-2)

        return integral_map

    @staticmethod
    def calc(integral_map, bbox_2d, min_area=1):
        """
        Args:
            bbox_2d: shape(N, 4)
        """
        F = integral_map
        # be sure integral number first
        bbox_2d = bbox_2d.long()
        bbox_2d[:, ::2].clamp_(min=0, max=F.shape[3] - 1)
        bbox_2d[:, 1::2].clamp_(min=0, max=F.shape[2] - 1)

        xmin = bbox_2d[:, 0]
        ymin = bbox_2d[:, 1]
        xmax = bbox_2d[:, 2]
        ymax = bbox_2d[:, 3]
        area = (ymax - ymin) * (xmax - xmin)

        # import ipdb
        # ipdb.set_trace()
        area_filter = area < min_area
        area[area_filter] = 1
        inds_filter = torch.nonzero(area_filter)[:, 0]
        # F[:, :, inds_filter, inds_filter] = 0

        res = (F[:, :, ymin, xmin] + F[:, :, ymax, xmax] - F[:, :, ymin, xmax]
               - F[:, :, ymax, xmin]) / area.type_as(F)
        res[:, :, inds_filter] = 0

        return res
