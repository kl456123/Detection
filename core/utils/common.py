# -*- coding: utf-8 -*-
import torch
import sys
import torch.nn as nn
import copy
import numpy as np


def build_class(config, registry, *args, **kwargs):
    if 'type' not in config:
        raise ValueError('config has no type, it can not be builded')
    class_type = config['type']
    if class_type not in registry:
        raise TypeError("unknown {} type {}!".format(registry.name,
                                                     class_type))
    registered_class = registry[class_type]
    # use config to build it
    return registered_class


def build(config, registry, *args, **kwargs):
    registered_class = build_class(config, registry, *args, **kwargs)

    # use config to build it
    return registered_class(config, *args, **kwargs)


def change_lr(lr, optimizer):
    for param_group in optimizer.param_groups:
        # used for scheduler
        param_group['initial_lr'] = lr


def to_cuda(target):
    if isinstance(target, list):
        return [to_cuda(e) for e in target]
    elif isinstance(target, dict):
        return {key: to_cuda(target[key]) for key in target}
    elif isinstance(target, np.ndarray):
        return to_cuda(torch.from_numpy(target))
    elif isinstance(target, torch.Tensor):
        return target.cuda()
    else:
        # dont change
        return target


def loss_dict_to_str(loss_dict):
    res_str = ""
    for key, val in loss_dict.items():
        res_str += "{}: {:.4f}\t".format(key, val.item())
    return res_str


class MyParallel(nn.DataParallel):
    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        return getattr(self.module, name)


class Stats(object):
    def __init__(self):
        super().__init__()
        self.stats = None

    # def collect_from_model(self, model):
    # return [
    # copy.deepcopy(target_generator.stats)
    # for target_generator in model.target_generators
    # ]

    def update_stats(self, stats):
        # stats = self.collect_from_model(model)
        if self.stats is None:
            self.stats = stats
        else:
            assert len(stats) == len(self.stats)
            for ind, stat in enumerate(stats):
                self.stats[ind] = self.merge_stats(self.stats[ind], stats[ind])

    def merge_stats(self, stats1, stats2):
        total_keys = set(stats1.keys()).union(set(stats2.keys()))
        stats = {}
        for key in total_keys:
            val1 = stats1[key].sum(dim=0, keepdim=True)
            val2 = stats2[key].sum(dim=0, keepdim=True)
            assert val1.shape[-1] == 2
            assert val2.shape[-1] == 2
            stats[key] = val1 + val2
        return stats

    def get_summary_dict(self):
        summary_dict = {}
        for idx, stat in enumerate(self.stats):
            for key in stat:
                value = stat[key][0]
                # tensor here
                summary_dict[key + '_' +
                             str(idx)] = value[0].float() / value[1].float()
        return summary_dict

    def __repr__(self):
        total_str = ""
        for stat in self.stats:
            for key in stat:
                value = stat[key][0]
                total_str += '{}: {}/{}/{:.4f}\t'.format(
                    key, value[0].item(), value[1].item(),
                    value[0].item() / value[1].item())
        return total_str

    def clear_stats(self):
        self.stats = None


# def make_parallel(module):
# return MyParallel(module)


def compile_cxx():
    in_path = '/node01_data5/'
    out_path = '/node01/jobs/io/out/xiongliang/'
    import os
    # command = 'cd /node01_data5/standard/BDD100k/labels && cp /node01/jobs/io/out/xiongliang/bdd100k_labels_images_train.json .'
    #  for file in os.listdir('/node01_data5/standard/BDD100k/labels'):
    #  print(file)
    #  asdg
    python = '/node01/jobs/io/env/pytorch1.0/bin/python'
    command = 'cd ./lib && {} setup.py build develop && rm build -rf && cd ..'.format(
        python)
    os.system(command)


instance_info_dict = {}


def InstanceInfo(object):
    def __init__(self, instance_info_dict):
        for attr_name in instance_info_dict:
            self.add_field_by_name()

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


def Assigner(object):
    pass
