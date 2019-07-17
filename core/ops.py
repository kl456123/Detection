# -*- coding: utf-8 -*-
import torch
from utils import image_utils
import numpy as np


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


def cylinderize(tensor, p2, radus=None):
    """
    Args:
        tensor: shape(NCHW)
        p2: shape(N, 3, 4)
    Returns:
        cylinder_tensor: shape(NCHW)
    """
    device = tensor.device
    N, C, H, W = tensor.shape

    tensor = tensor.permute(0, 2, 3, 1).view(N, H, W, -1)
    row = torch.arange(H).to(device)
    col = torch.arange(W).to(device)
    x, y = meshgrid(col, row)
    x = x.view(-1)
    y = y.view(-1)
    xy = torch.stack([x, y], dim=-1).float()

    cylinder_tensor = torch.zeros_like(tensor)

    f = p2[:, 0, 0]
    u0 = p2[:, 0, 2]
    if radus is None:
        radus = W / (
            image_utils.arctan(f * W / (u0 * (u0 - W) + f * f) + 0.5 * np.pi))

    cylinder_tensor = []
    for batch_ind in range(N):
        uv = image_utils.cylinder_to_plane(xy, p2[batch_ind], radus[batch_ind])
        cylinder_tensor.append(
            image_utils.torch_batch_bilinear_intep(tensor[batch_ind], uv).view(
                H, W, -1))

    cylinder_tensor = torch.stack(cylinder_tensor, dim=0)

    return cylinder_tensor.permute(0, 3, 1, 2)
