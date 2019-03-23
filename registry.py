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
BBOX_CODER = Registry('bbox_coder')

# similarity_calc
SIMILARITY_CALC = Registry('similarity_calc')

# matcher
MATCHER = Registry('matcher')
