# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import torch


class SimilarityCalc(ABC):
    def __init__(self, config):
        pass

    #  def compare_batch(self, anchor, gt_boxes, num_instances):
    #  """
    #  Args:
    #  anchor: shape(N, M, 4)
    #  gt_boxes: shape(N, K, 4)
    #  Returns:
    #  match_quality_matrix: shape(N, M, K)
    #  """
    #  batch_size = gt_boxes.shape[0]
    #  M = anchor.shape[1]
    #  K = gt_boxes.shape[1]
    #  device = gt_boxes.device
    #  match_quality_matrix = torch.zeros(
    #  (batch_size, M, K)).float().to(device)
    #  for batch_ind in range(batch_size):
    #  anchor_per_image = anchor[batch_ind]
    #  gt_boxes_per_img = gt_boxes[batch_ind][:num_instances[batch_ind]]
    #  match_quality_matrix[batch_ind, :, :num_instances[
    #  batch_ind]] = self.compare(anchor_per_image, gt_boxes_per_img)
    #  return match_quality_matrix

    @abstractmethod
    def compare_batch(self, anchor, gt_boxes):
        pass
