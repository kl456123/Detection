# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import torch


class Matcher(ABC):
    def __init__(self, match_config):
        pass
        # pass
        # information is generated during matching
        # self._assigned_overlaps = None
        # self._assigned_overlaps_batch = None

    @abstractmethod
    def match(self, match_quality_matrix, thresh):
        pass

    def match_batch(self, match_quality_matrix_batch, num_instances, thresh):
        """
        batch version of assign function
        Args:
            match_quality_matrix_batch: shape(N,num_boxes,num_gts)
        Returns:
            assignments: shape(N, M)
            overlaps: shape(N, M)
        """
        batch_size = match_quality_matrix_batch.shape[0]
        assignments = []
        overlaps = []

        for i in range(batch_size):
            # shape(K)
            assignments_per_img, max_overlaps_per_img = self.match(
                match_quality_matrix_batch[i, :, :num_instances[i]], thresh)
            assignments.append(assignments_per_img)
            overlaps.append(max_overlaps_per_img)

        # shape(N,num_boxes)
        assignments = torch.stack(assignments)
        # shape(N,num_boxes)
        overlaps = torch.stack(overlaps)
        # self._assigned_overlaps_batch = overlaps

        # self._assignments = assignments
        return assignments, overlaps
