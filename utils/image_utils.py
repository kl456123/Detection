# -*- coding: utf-8 -*-
"""
    some preprocessing operators
"""
import torch


def drawGaussian(pt, image_shape, sigma=2):
    """
    Args:
        pt: shape(N, M, K, 2)
        sigma: scalar 2 or 3 in common case
        image_shape: shape(2)
    Returns:
        keypoint_heatmap: shape(N, M, K, S, S)
    """

    tmpSize = 3 * sigma
    # Check that any part of the gaussian is in-bounds
    N, M, K = pt.shape[:3]
    pt = pt.view(-1, 2).long()
    ul = torch.stack([pt[..., 0] - tmpSize, pt[..., 1] - tmpSize], dim=-1)
    br = torch.stack(
        [pt[..., 0] + tmpSize + 1, pt[..., 1] + tmpSize + 1], dim=-1)

    cond = (ul[..., 0] >= image_shape[1]) | (ul[..., 1] >= image_shape[0]) | (
        br[..., 0] < 0) | (br[..., 1] < 0)

    # Generate gaussian
    size = 2 * tmpSize + 1
    x = torch.arange(size, dtype=torch.float)
    y = x[:, None]
    x0 = y0 = size // 2
    sigma = size / 4.0
    # The gaussian is not normalized, we want the center value to equal 1
    g = torch.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)).float().to(
        pt.device)

    # Usable gaussian range
    g_x_start = (-ul[..., 0]).clamp(min=0)
    g_x_end = br[..., 0].clamp(max=image_shape[1]) - ul[..., 0]
    g_y_start = (-ul[..., 1]).clamp(min=0)
    g_y_end = br[..., 1].clamp(max=image_shape[0]) - ul[..., 1]
    # g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    # g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x_start = ul[..., 0].clamp(min=0)
    img_y_start = ul[..., 1].clamp(min=0)
    img_x_end = br[..., 0].clamp(max=image_shape[1])
    img_y_end = br[..., 1].clamp(max=image_shape[0])
    # img_x = max(0, ul[0]), min(br[0], img.shape[1])
    # img_y = max(0, ul[1]), min(br[1], img.shape[0])

    # assign from gaussian distribution
    # img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    h, w = image_shape
    img = torch.zeros(N, M, K, h, w).type_as(pt).float()

    g_index = torch.nonzero(g > 0).unsqueeze(-1)
    g_cond = (g_index[:, 1] >= g_x_start) & (g_index[:, 0] >= g_y_start) & (
        g_index[:, 1] < g_x_end) & (g_index[:, 0] < g_y_end)
    img_index = torch.nonzero(img.view(-1, h, w)[0] > -1).unsqueeze(-1)
    img_cond = (img_index[:, 1] >= img_x_start) & (
        img_index[:, 0] >= img_y_start) & (img_index[:, 1] < img_x_end) & (
            img_index[:, 0] < img_y_end)
    img_cond = img_cond.transpose(0, 1).view(N, M, K, h, w)
    g_cond = g_cond.transpose(0, 1).view(N, M, K, g.shape[0], g.shape[1])
    # import ipdb
    # ipdb.set_trace()
    img[img_cond] = g.expand_as(g_cond)[g_cond]

    return img
