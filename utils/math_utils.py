# -*- coding: utf-8 -*-
"""
Math function
"""
import math
import torch


def gaussian2d(pos, center, sigma=1):
    pos = (pos - center) / sigma.unsqueeze(-1)
    # A = 1 / (sigma * math.sqrt(2 * math.pi))
    G = torch.exp(-0.5 *
                  (pos[..., 0] * pos[..., 0] + pos[..., 1] * pos[..., 1]))
    G[G > 1] = 1
    return G
