# -*- coding: utf-8 -*-

from utils.registry import BBOX_CODERS
from core import constants
import torch


@BBOX_CODERS.register(constants.KEY_DIMS)
class MeanCoder(object):
    @staticmethod
    def encode_batch(gt_dims, mean_dims):
        return gt_dims - mean_dims

    @staticmethod
    def decode_batch(pred_dims, mean_dims, probs):
        """
        Args:
            pred_dims: shape(N, M, 3)
            mean_dims: shape(N, num_classes, 3)
            probs: shape(N, M, num_classes)
        """
        _, probs_argmax = torch.max(probs, dim=-1)

        bg_dim = torch.zeros_like(mean_dims[:, -1:, :])
        mean_dims = torch.cat([bg_dim, mean_dims], dim=1)
        mean_dims = mean_dims[0][probs_argmax]

        return pred_dims + mean_dims
