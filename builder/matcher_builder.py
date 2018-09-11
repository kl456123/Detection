# -*- coding: utf-8 -*-

from core.matchers.bipartitle_matcher import BipartitleMatcher
from core.matchers.argmax_matcher import ArgmaxMatcher
from core.matchers.scale_matcher import ScaleMatcher


def build(matcher_config):
    matcher_type = matcher_config['type']
    if matcher_type == 'argmax':
        return ArgmaxMatcher(matcher_config)
    elif matcher_type == 'bipartitle':
        return BipartitleMatcher(matcher_config)
    elif matcher_type == 'scale':
        return ScaleMatcher(matcher_config)
    else:
        raise ValueError("unknown matcher type {}!".format(matcher_type))
