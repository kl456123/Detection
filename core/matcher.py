# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import torch


class Matcher(ABC):
    def __init__(self):
        # information is generated during matching
        self._assigned_overlaps = None
        self._assigned_overlaps_batch = None

    @property
    def assigned_overlaps(self):
        return self._assigned_overlaps

    @property
    def assigned_overlaps_batch(self):
        return self._assigned_overlaps_batch

    @abstractmethod
    def match(self, match_quality_matrix, thresh):
        pass

    def match_batch(self, match_quality_matrix_batch, thresh):
        """
        batch version of assign function
        Args:
            match_quality_matrix_batch: shape(N,num_boxes,num_gts)
        """
        batch_size = match_quality_matrix_batch.shape[0]
        assignments = []
        overlaps = []

        for i in range(batch_size):
            # shape(K)
            assignments_per_img = self.match(match_quality_matrix_batch[i],
                                             thresh)
            assignments.append(assignments_per_img)
            overlaps.append(self._assigned_overlaps)

        # shape(N,num_boxes)
        assignments = torch.stack(assignments)
        # shape(N,num_boxes)
        overlaps = torch.stack(overlaps)
        self._assigned_overlaps_batch = overlaps

        return assignments
