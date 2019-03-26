# -*- coding: utf-8 -*-

# import models
from models.feature_extractors import *
from models.backbones import *
from models.detectors import *
from models.losses import *

# import all other components
from matchers import *
from target_assigners import *
from similarity_calcs import *
from samplers import *
from bbox_coders import *
from anchor_generators import *

# import all components about datasets
from data.datasets import *
from data.transforms import *

# import some components about training
from solvers import *

# at last import registry
from utils import registry


def build(config, registry, *args, **kwargs):
    if 'type' not in config:
        raise ValueError('config has no type, it can not be builded')
    class_type = config['type']
    if class_type not in registry:
        raise TypeError("unknown {} type {}!".format(
            registry.name, class_type))
    registered_class = registry[class_type]
    # use config to build it
    return registered_class(config, *args, **kwargs)


def build_backbone(config):
    return build(config, registry.BACKBONES)


def build_matcher(config):
    return build(config, registry.MATCHERS)


def build_anchor_generator(config):
    return build(config, registry.ANCHOR_GENERATORS)


def build_sampler(config):
    return build(config, registry.SAMPLERS)


def build_dataloader(config):
    batch_size = config['batch_size']
    shuffle = config['shuffle']
    num_workers = config['num_workers']

    # build dataset first
    dataset_config = config['dataset_config']
    dataset = build_dataset(dataset_config)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers)
    return dataloader


def build_dataset(config):
    # build transform first
    transform_config = config['transform_config']
    transform = build_transform(transform_config)

    return build(config, registry.DATASETS, transform)


def build_similarity_calc(config):
    return build(config, registry.SIMILARITY_CALCS)


def build_transform(config):
    """
    building transform is different from all above,
    it needs combine components of transforms
    """
    transforms = []
    for trans_config in config:
        transforms.append(build(trans_config, registry.TRANSFORMS))
    return trans.Compose(transforms)


def build_optimizer(optimizer_config, model):
    build(optimizer_config, registry.OPTIMIZERS, model)


def build_scheduler(scheduler_config):
    pass
