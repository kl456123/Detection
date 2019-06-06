# -*- coding: utf-8 -*-

from collections import OrderedDict
import torch
from core import constants


class Instance3D(object):
    FIELDS = [
        constants.KEY_BOXES_2D, constants.KEY_CLASSES, constants.KEY_DIMS,
        constants.KEY_ORIENTS, constants.KEY_BOXES_3D
    ]

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


instance_info_dict = {}


def InstanceInfo(object):
    def __init__(self, instance_info_dict):
        for attr_name in instance_info_dict:
            self.add_field_by_name(attr_name, coders)

    def add_field_by_name(self, name, coders=[]):
        """
        Note that one name for multiple coders
        """
        attr = Attr(name)
        # for coder in coders:
        # attr.add_coder(coder)

        self.add_field_by_attr(self, attr)

    def add_field_by_attr(self, attr):
        setattr(self, attr.name, attr)


def Coder(object):
    pass


def Attr(object):
    def __init__(self, name):
        self.name = name
        self._set_assigner(name)
        self._set_coders(name)

    def _set_assigner(self, name):
        import coders
        config = {'type': name}
        self._assigner = coders.build(config)

    # def add_coder(self, coder):
    # pass

    def _set_coders(self, name):
        import bbox_coders

        config = {'type': name}
        self._coders = bbox_coders.build(config)


def Loss(dict):
    def update_loss_units(self, name, preds=None, weights=None, targets=None):
        self[name].update({
            'pred': preds,
            'weight': weights,
            'target': targets
        })



