# -*- coding: utf-8 -*-
import torch
import math


def meshgrid(a, b):
    """
    Args:
        a: tensor of shape(N)
        b: tensor of shape(M)
    Returns:
        two tensor of shape(M,N) and shape (M,N)
    """
    a_ = a.repeat(b.numel())
    b_ = b.repeat(a.numel(), 1).t().contiguous().view(-1)
    return a_, b_


def get_angle(sin, cos):
    """
        Args:
            sin: shape(N,num_bins)
        """
    sin = sin.detach()
    cos = cos.detach()
    norm = torch.sqrt(sin * sin + cos * cos)
    sin /= norm
    cos /= norm

    # range in [-pi, pi]
    theta = torch.asin(sin)
    cond_pos = (cos < 0) & (sin > 0)
    cond_neg = (cos < 0) & (sin < 0)
    theta[cond_pos] = math.pi - theta[cond_pos]
    theta[cond_neg] = -math.pi - theta[cond_neg]
    return theta
