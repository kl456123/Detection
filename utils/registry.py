# -*- coding: utf-8 -*-

from core.registry import Registry

# classic models used as backbones
BACKBONES = Registry('backbones')

# modules used for extracting features
FEATURE_EXTRACTORS = Registry('feature_extractors')

LOSSES = Registry('losses')

# whole models used as detectors
DETECTORS = Registry('detectors')

# bbox coder(transform bbox format)
BBOX_CODERS = Registry('bbox_coder')

# similarity_calc
SIMILARITY_CALCS = Registry('similarity_calcs')

# matcher
MATCHERS = Registry('matchers')

# anchor_generators
ANCHOR_GENERATORS = Registry('anchor_generators')

# dataset and dataloaders
DATASETS = Registry('datasets')
DATALOADERS = Registry('dataloaders')

# samplers
SAMPLERS  = Registry('samplers')

# transforms
TRANSFORMS = Registry('transforms')

############################
########## TRAIN ###########
############################

# optimizers
OPTIMIZERS = Registry('optimizers')

# schedulers
SCHEDULERS = Registry('schedulers')
