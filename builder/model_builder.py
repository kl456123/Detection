# -*- coding: utf-8 -*-

from models import *
from register import DETECTORS


def build(model_config, training):
    net_type = model_config['net']
    detector = DETECTORS[net_type](model_config)
    if training:
        detector.train()
    else:
        detector.eval()
