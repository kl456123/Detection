# -*- coding: utf-8 -*-
"""
    some preprocessing operators
"""
import torch
import numpy as np


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


def bilinear_intep(image, xy):
    """
    Args:
        image: shape(H, W, 3)
        xy: just refers to (x, y)
    """
    x = xy[0]
    y = xy[1]
    h, w = image.shape[:2]
    if x >= w - 1 or y >= h - 1 or x < 0 or y < 0:
        return [255, 255, 255]
    xmin = int(x)
    xmax = xmin + 1
    ymin = int(y)
    ymax = ymin + 1
    a = xmax - x
    b = ymax - y
    pixel_value = (1 - a) * (1 - b) * image[ymin, xmin] + a * (
        1 - b) * image[ymax, xmin] + (
            1 - a) * b * image[ymin, xmax] + a * b * image[ymax, xmax]
    return pixel_value


def batch_bilinear_intep(image, xypoints):
    """
    Args:
        image: shape(H,W,3)
        xypoints: (N,2)
    Returns:
        pixel_value: shape(N, 3)
    """
    # import ipdb
    # ipdb.set_trace()
    x = xypoints[:, 0]
    y = xypoints[:, 1]
    h, w = image.shape[:2]
    outside_cond = (x >= w - 1) | (y >= h - 1) | (x < 0) | (y < 0)
    x = x[~outside_cond]
    y = y[~outside_cond]
    xmin = np.floor(x).astype(np.int)
    xmax = np.ceil(x).astype(np.int)
    ymin = np.floor(y).astype(np.int)
    ymax = np.ceil(y).astype(np.int)
    a = xmax - x
    b = ymax - y
    a = a[:, None]
    b = b[:, None]
    pixel_value = (1 - a) * (1 - b) * image[ymin, xmin] + a * (
        1 - b) * image[ymax, xmin] + (
            1 - a) * b * image[ymin, xmax] + a * b * image[ymax, xmax]
    total_pixel_value = np.zeros((xypoints.shape[0], 3))
    total_pixel_value[np.where(~outside_cond)[0]] = pixel_value

    return total_pixel_value


def torch_batch_bilinear_intep(image, xypoints):
    """
    Args:
        image: shape(H,W,3)
        xypoints: (N,2)
    Returns:
        pixel_value: shape(N, 3)
    """
    # import ipdb
    # ipdb.set_trace()
    x = xypoints[:, 0]
    y = xypoints[:, 1]
    h, w = image.shape[:2]
    outside_cond = (x >= w - 1) | (y >= h - 1) | (x < 0) | (y < 0)
    x[outside_cond] = 0
    y[outside_cond] = 0
    # x = x[~outside_cond]
    # y = y[~outside_cond]
    xmin = torch.floor(x).long()
    xmax = torch.ceil(x).long()
    ymin = torch.floor(y).long()
    ymax = torch.ceil(y).long()
    a = xmax.float() - x
    b = ymax.float() - y
    a = a[:, None]
    b = b[:, None]
    pixel_value = (1 - a) * (1 - b) * image[ymin, xmin] + a * (
        1 - b) * image[ymax, xmin] + (
            1 - a) * b * image[ymin, xmax] + a * b * image[ymax, xmax]
    # total_pixel_value = torch.zeros((xypoints.shape[0],
    # image.shape[-1])).type_as(image)
    # total_pixel_value[torch.nonzero(~outside_cond).view(-1)] = pixel_value
    pixel_value[torch.nonzero(outside_cond).view(-1)] = 0

    return pixel_value


def cylinder_to_plane(xy, p2, radus):
    """
    Args:
        xy: shape(N, 2)
    """
    u0 = p2[0, 2]
    f = p2[0, 0]
    x = xy[:, 0]
    u = u0 - f * ((u0 / f - tan(x / radus)) / (1 + u0 / f * tan(x / radus)))
    v = xy[:, 1]

    return stack([u, v], axis=-1)


def plane_to_cylinder(uv, p2, radus):
    #  import ipdb
    #  ipdb.set_trace()
    u0 = p2[0, 2]
    f = p2[0, 0]
    u = uv[:, 0]
    x = arctan2(u * f , (f * f + (u0 - u) * u0)) * radus
    y = uv[:, 1]

    return stack([x, y], axis=-1)


def stack(x, axis):
    if isinstance(x[0], torch.Tensor):
        return torch.stack(x, dim=axis)
    else:
        return np.stack(x, axis=axis)


def arctan(x):
    """
    convert from [-0.5pi, 0.5pi] to [0, np.pi]
    """
    if isinstance(x, torch.Tensor):
        x = torch.atan(x.float())
    else:
        x = np.arctan(x)
    x[x < 0] = x[x < 0] + 0.5 * np.pi
    return x


def arctan2(y, x):
    if isinstance(x, torch.Tensor):
        x = torch.atan2(y, x)
    else:
        x = np.arctan2(y, x)
    return x


def tan(x):
    if isinstance(x, torch.Tensor):
        return torch.tan(x.float())
    return np.tan(x)


def cylinder_project(image, p2, radus=None):
    h, w = image.shape[:2]
    # radius = p2[0, 0]

    # calc radius

    f = p2[0, 0]
    u0 = p2[0, 2]
    cylinder_image = np.zeros_like(image)
    cylinder_image[...] = 255
    if radus is None:
        radus = w / (arctan(f * w / (u0 * (u0 - w) + f * f) + 0.5 * np.pi))

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    x = x.reshape(-1)
    y = y.reshape(-1)

    #  u = u0 - f * ((u0 / f - np.tan(x / radus)) /
    #  (1 + u0 / f * np.tan(x / radus)))
    #  v = y
    #  uv = np.stack([u, v], axis=-1)
    xy = np.stack([x, y], axis=-1)
    uv = cylinder_to_plane(xy, p2, radus)
    cylinder_image[y, x] = batch_bilinear_intep(image, uv)
    return cylinder_image


if __name__ == '__main__':

    import cv2
    import os
    import sys
    p2 = np.asarray([
        7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02,
        4.575831000000e+01, 0.000000000000e+00, 7.070493000000e+02,
        1.805066000000e+02, -3.454157000000e-01, 0.000000000000e+00,
        0.000000000000e+00, 1.000000000000e+00, 4.981016000000e-03
    ]).reshape(3, 4)
    saved_dir = 'results/cylinder_images'
    image_dir = '/data/object/training/image_2/'
    for ind, image_name in enumerate(sorted(os.listdir(image_dir))):
        # import ipdb
        # ipdb.set_trace()
        image_path = os.path.join(image_dir, image_name)
        image = cv2.imread(image_path)
        cylinder_image = cylinder_project(image, p2, radus=None)
        saved_path = os.path.join(saved_dir, image_name)
        cv2.imwrite(saved_path, cylinder_image)
        sys.stdout.write('\r{}'.format(ind))
        sys.stdout.flush()
