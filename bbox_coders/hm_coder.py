# -*- coding: utf-8 -*-

import torch

from utils.registry import BBOX_CODERS
from core import constants
from utils import geometry_utils
from utils import image_utils

from core.utils import format_checker

# heatmap_size = [56, 56]


@BBOX_CODERS.register(constants.KEY_KEYPOINTS_HEATMAP)
class KeyPointCoder(object):
    # def __init__(self, coder_config):
    # self.heatmap_size = coder_config['size']
    resolution = 28

    @staticmethod
    def encode_batch(bboxes, keypoints_pairs):
        """
        Note that it is not necessary to convert maps to spatial format
        Args:
            keypoints: shape(N,M,K,3), here M refers to num of instances, K refers to num of joints
            rois: shape(N,M, 4)
        Returns:
            keypoint_heatmap: shape(N, M, K, 2) (pos, vis)
        """
        # import ipdb
        # ipdb.set_trace()
        N, M = keypoints_pairs.shape[:2]
        keypoints_pairs = keypoints_pairs.view(N, M, -1, 3)
        num_joints = keypoints_pairs.shape[-2]
        # import ipdb
        # ipdb.set_trace()
        keypoints = keypoints_pairs[..., :2]
        visibility = keypoints_pairs[..., 2]

        # spatial_dims = heatmap_size[0] * heatmap_size[1]
        # heatmaps_shape = (N, M, num_joints, spatial_dims)
        # heatmaps = torch.zeros(heatmaps_shape).type_as(bboxes)

        peak_pos, peak_offsets = KeyPointCoder._calculate_peak_pos(
            bboxes, keypoints)

        resolution = KeyPointCoder.resolution
        hm = image_utils.drawGaussian(
            peak_pos, (resolution, resolution), sigma=2)
        hm = hm.view(N, M, num_joints, -1)
        # filter inside of the window
        # inside_filter = (peak_pos[:, :, 0] < heatmap_size[1]) & (
        # peak_pos[:, :, 1] < heatmap_size[0]) & (peak_pos[:, :, 0] >= 0) & (
        # peak_pos[:, :, 1] >= 0)

        # assign peak to heatmaps
        # heatmaps = heatmaps.view(-1, spatial_dims)
        # peak_pos = peak_pos.view(-1, 2)
        # inside_filter = inside_filter.view(-1)

        # # import ipdb
        # # ipdb.set_trace()
        # peak_pos = peak_pos[..., 1] * resolution + peak_pos[..., 0]

        # peak_weights = torch.zeros_like(peak_offsets).type_as(peak_offsets)
        # peak_weights[inside_filter] = 1
        # peak = torch.stack([peak_offsets, peak_weights], dim=-1)
        # peak = peak.view(-1, 4 * 2).float()
        # import ipdb
        # ipdb.set_trace()
        encoded_keypoints = torch.cat([hm, visibility.unsqueeze(-1)], dim=-1)
        return encoded_keypoints.view(N, M, -1)

    @staticmethod
    def _calculate_peak_pos(bboxes, keypoints):
        # convert to (w,h)
        resolution = KeyPointCoder.resolution
        heatmap_size = torch.tensor((resolution, resolution)).type_as(bboxes)

        bboxes_xywh = geometry_utils.torch_xyxy_to_xywh(bboxes)
        wh = bboxes_xywh[..., 2:].unsqueeze(-2)
        bboxes = bboxes.unsqueeze(dim=-2)

        bboxes_w = bboxes[..., 2] - bboxes[..., 0]
        bboxes_h = bboxes[..., 3] - bboxes[..., 1]
        # note that (w,h) here
        bboxes_dim = torch.stack([bboxes_w, bboxes_h], dim=-1)

        # shape(N,K,2)
        peak_pos_norm = (keypoints - bboxes[..., :2]) / bboxes_dim
        peak_pos_float = (peak_pos_norm * heatmap_size)

        # make sure all pos in the inner of bbox
        # if not, use the nearest pos instead
        peak_pos_int = peak_pos_float.floor().clamp(min=0, max=resolution - 1)

        # offset between peak_pos_int and peak_pos_float
        peak_offsets = (peak_pos_float - peak_pos_int) / wh

        return peak_pos_int, peak_offsets

    @staticmethod
    def decode_batch(bboxes, keypoint_heatmap, pixel_offsets=0.5):
        """
        Args:
            rois: shape(N, M, 4)
            keypoint_heatmap: shape(N, M, K, m*m)
        Returns:
            keypoints: shape(N,K,2)
        """
        # import ipdb
        # ipdb.set_trace()
        resolution = KeyPointCoder.resolution
        N, M = keypoint_heatmap.shape[:2]
        keypoint_heatmap = keypoint_heatmap.view(N, M, 8, -1)
        _, peak_pos = keypoint_heatmap.max(dim=-1)

        # select offset preds from heatmap
        # keypoint_heatmap = keypoint_heatmap.view(
            # N * M * 8, -1)
        # row = torch.arange(peak_pos.numel()).type_as(peak_pos)
        # offsets = keypoint_heatmap[row, peak_pos.view(-1)].view(N, M, 8,
                                                                # 3)[..., 1:]

        peak_pos_y = peak_pos / resolution
        peak_pos_x = peak_pos % resolution
        peak_pos = torch.stack([peak_pos_x, peak_pos_y], dim=-1).float()

        # new_heatmap_size = (heatmap_size[1], heatmap_size[0])
        new_heatmap_size = torch.tensor((resolution,
                                         resolution)).type_as(peak_pos)
        peak_pos_norm = (peak_pos + pixel_offsets) / new_heatmap_size

        bboxes_xywh = geometry_utils.torch_xyxy_to_xywh(bboxes)
        wh = bboxes_xywh[..., 2:].unsqueeze(-2)
        bboxes = bboxes.unsqueeze(-2)
        bboxes_w = bboxes[..., 2] - bboxes[..., 0]
        bboxes_h = bboxes[..., 3] - bboxes[..., 1]
        # note that (w,h) here
        bboxes_dim = torch.stack([bboxes_w, bboxes_h], dim=-1)

        keypoints = peak_pos_norm * bboxes_dim + bboxes[..., :2]

        # keypoints + offsets
        # keypoints = keypoints + offsets * wh
        return keypoints
