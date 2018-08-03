# -*- coding: utf-8 -*-

from core.matcher import Matcher


class BipartitleMatcher(Matcher):
    def match(self, similarity_matrix):
        """
        similarity_matrix: shape(num_anchors,num_gt_boxes),can be iou or other critation
        """
        pass
