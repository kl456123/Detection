# -*- coding: utf-8 -*-

from . import *
from core.utils.common import build
from utils.registry import SAMPLERS


def bulid(config):
    return build(config, SAMPLERS)
