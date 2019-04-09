# -*- coding: utf-8 -*-

from collections import OrderedDict
import torch
from core import constants


class Instance3D(object):
    FIELDS = [constants.KEY_BOXES_2D, constants.KEY_CLASSES,
              constants.KEY_DIMS, constants.KEY_ORIENTS,
              constants.KEY_BOXES_3D]

    def __init__(self):
        self._fields = OrderedDict()

    def add_field(self, key, value=None):
        self._fields[key] = value

    def filter_field(self, field_filter):
        fields = self._fields
        for key in fields:
            fields[key] = fields[key][field_filter]

    def squeeze(self):
        items = []
        for value in self._fields.items():
            items.append(value)
        return torch.cat(items, dim=-1)

    @staticmethod
    def from_dict(prediction_dict):
        instance_3d = Instance3D()
        for field in Instance3D.FIELDS:
            if field in prediction_dict:
                instance_3d.add_field(field, prediction_dict[field])
        return instance_3d

    def __len__(self):
        first = next(iter(self._fields))
        if first is None:
            return 0
        else:
            assert isinstance(first, torch.Tensor)
            return first.shape[0]


class Instance2D(object):
    pass
