# -*- coding: utf-8 -*-

from core.similarity_calc.scale_similarity_calc import ScaleSimilarityCalc
from core.similarity_calc.center_similarity_calc import CenterSimilarityCalc


def build(similarity_calc_config):
    similarity_calc_type = similarity_calc_config['type']
    if similarity_calc_type == 'scale':
        return ScaleSimilarityCalc()
    elif similarity_calc_type == 'center':
        return CenterSimilarityCalc()
    else:
        raise ValueError('unsupported type of similarity_calc!')
