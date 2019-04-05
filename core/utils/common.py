# -*- coding: utf-8 -*-
import torch
import sys
import torch.nn as nn
import copy


def build_class(config, registry, *args, **kwargs):
    if 'type' not in config:
        raise ValueError('config has no type, it can not be builded')
    class_type = config['type']
    if class_type not in registry:
        raise TypeError(
            "unknown {} type {}!".format(registry.name, class_type))
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
    elif isinstance(target, torch.Tensor):
        return target.cuda()
    else:
        # dont change
        return target


def print_loss(loss_dict):
    print_num = 0
    for key, val in loss_dict.items():
        if print_num % 3 == 0:
            sys.stdout.write("\t\t\t")
        sys.stdout.write("{}: {:.4f}\t".format(key, val.mean().item()))
        print_num += 1
        if print_num % 3 == 0:
            sys.stdout.write("\n")
    if print_num % 3:
        sys.stdout.write("\n")


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

    def collect_from_model(self, model):
        return [
            copy.deepcopy(target_generator.stats)
            for target_generator in model.target_generators
        ]

    def update_stats(self, model):
        stats = self.collect_from_model(model)
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
            assert len(stats1[key]) == 2
            assert len(stats2[key]) == 2
            value1 = stats1.get(key, (0, 0))
            value2 = stats2.get(key, (0, 0))
            stats[key] = (value1[0] + value2[0], value2[1] + value1[1])
        return stats

    def get_summary_dict(self):
        summary_dict = {}
        for stat in self.stats:
            for key in stat:
                value = stat[key]
                # tensor here
                summary_dict[key] = value[0].float() / value[1].float()
        return summary_dict

    def __repr__(self):
        total_str = []
        for stat in self.stats:
            for key in stat:
                value = stat[key]
                total_str.append(
                    '\t\t\t{}: {}/{}/{:.4f}'.format(key, value[0].item(
                    ), value[1].item(), value[0].item() / value[1].item()))
        return '\n'.join(total_str)

    def clear_stats(self):
        self.stats = None


# def make_parallel(module):
# return MyParallel(module)
