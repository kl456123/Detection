# -*- coding: utf-8 -*-

from .build import register_all_backbones
from .build import build_backbone, build_backbone_path

register_all_backbones()
__all__ = ['build_backbone', 'build_weights_fname']
