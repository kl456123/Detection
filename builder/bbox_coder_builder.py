# -*- coding: utf-8 -*-

from core.bbox_coders.center_coder import CenterCoder
from core.bbox_coders.bbox_3d_coder import BBox3DCoder
from core.bbox_coders.geometry_v3_coder import GeometryV3Coder
from core.bbox_coders.geometry_v2_coder import GeometryV2Coder
from core.bbox_coders.geometry_v1_coder import GeometryV1Coder
from core.bbox_coders.oft_coder import OFTBBoxCoder
from core.bbox_coders.keypoint_coder import KeyPointCoder


def build(coder_config):
    coder_type = coder_config['type']
    if coder_type == 'center':
        return CenterCoder(coder_config)
    elif coder_type == 'geometry_v3':
        return GeometryV3Coder(coder_config)
    elif coder_type == 'geometry_v2':
        return GeometryV2Coder(coder_config)
    elif coder_type == 'geometry_v1':
        return GeometryV1Coder(coder_config)
    elif coder_type == 'bbox_3d':
        return BBox3DCoder(coder_config)
    elif coder_type == 'oft':
        return OFTBBoxCoder(coder_config)
    elif coder_type == 'keypoint':
        return KeyPointCoder(coder_config)
    else:
        raise ValueError('unknown type of bbox coder')
