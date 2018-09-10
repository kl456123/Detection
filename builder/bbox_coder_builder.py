# -*- coding: utf-8 -*-

from core.bbox_coders.center_coder import CenterCoder
from core.bbox_coders.scale_coder import ScaleCoder


def build(coder_config):
    coder_type = coder_config['type']
    if coder_type == 'scale':
        return ScaleCoder(coder_config)
    elif coder_type == 'center':
        return CenterCoder(coder_config)
    else:
        raise ValueError('unknown type of bbox coder')
