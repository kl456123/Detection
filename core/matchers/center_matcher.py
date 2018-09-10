# -*- coding: utf-8 -*-

#!/usr/bin/env python
# encoding: utf-8

from core.matcher import Matcher
import torch


class CenterMatcher(Matcher):
    def __init__(self, matcher_config):
        super().__init__()

    def match(self, match_quality_matrix, thresh):
        """
        match each bbox with gts
        if center of bbox is in the inner of gt box, bbox matchs with gt box

        Args:
            match_quality_matrix: usually overlaps is used
        """


