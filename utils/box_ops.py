# -*- coding: utf-8 -*-
import torch
import similarity_calcs
from utils import geometry_utils
import numpy as np

# from similarity_calcs.center_similarity_calc import CenterSimilarityCalc


def box2rois(bboxes):
    """
    Add img dim to each bbox
    Args:
        bboxes: shape(N, M, 4)
    Returns:
        bboxes: shape(N, M, 5)
    """
    batch_size = bboxes.shape[0]
    batch_idx = torch.arange(batch_size).view(batch_size, 1).expand(
        -1, bboxes.shape[1]).type_as(bboxes)
    rois = torch.cat((batch_idx.unsqueeze(-1), bboxes), dim=2)
    return rois


def clip_boxes(boxes, image_shape):
    """
    Args:
        boxes: shape(N, M, 4)
        image_shape: shape(N, 2)
    """
    boxes = boxes.clone()
    boxes[boxes < 0] = 0
    num_boxes = boxes.shape[1]
    boxes_x = boxes[:, :, ::2]
    boxes_y = boxes[:, :, 1::2]
    y_boundary = image_shape[:, 0].unsqueeze(1).unsqueeze(1).repeat(
        1, num_boxes, 2)
    x_boundary = image_shape[:, 1].unsqueeze(1).unsqueeze(1).repeat(
        1, num_boxes, 2)
    boxes_x[boxes_x > x_boundary] = x_boundary[boxes_x > x_boundary]
    boxes_y[boxes_y > y_boundary] = y_boundary[boxes_y > y_boundary]
    boxes[:, :, ::2] = boxes_x
    boxes[:, :, 1::2] = boxes_y

    return boxes


def np_clip_boxes(boxes, image_info):
    """
    Args:
        boxes: shape(N, 4) (x1,y1,x2,y2)
        image_info: shape(4,) (h,w, sh, sw)
    """
    #  import ipdb
    #  ipdb.set_trace()
    new_boxes = np.copy(boxes)
    new_boxes[...] = np.maximum(boxes, 0)
    # new_boxes[:, 0] = np.maximum(boxes[:, 0], 0)
    # new_boxes[:, 1] = np.maximum(boxes[:, 1], 0)
    new_boxes[:, 0] = np.minimum(new_boxes[:, 0], image_info[1])
    new_boxes[:, 1] = np.minimum(new_boxes[:, 1], image_info[0])

    new_boxes[:, 2] = np.minimum(new_boxes[:, 2], image_info[1])
    new_boxes[:, 3] = np.minimum(new_boxes[:, 3], image_info[0])
    return new_boxes


def window_filter(anchors, window, allowed_border=0):
    """
    Args:
        anchors: Tensor(N,4) ,xxyy format
        window: length-2 list (h,w)
    Returns:
        keep: Tensor(N,) bool type
    """
    keep = ((anchors[:, 0] >= -allowed_border) &
            (anchors[:, 1] >= -allowed_border) &
            (anchors[:, 2] < window[1] + allowed_border) &
            (anchors[:, 3] < window[0]) + allowed_border)
    return keep


def size_filter(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
    hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
    mask = (ws >= min_size) & (hs >= min_size)
    return mask


def area(boxes):
    """
    Get Area of batch bboxes
    Args:
        boxes: shape(N,M,4)
    """
    w = boxes[:, :, 2] - boxes[:, :, 0] + 1
    h = boxes[:, :, 3] - boxes[:, :, 1] + 1
    return w * h


def iou(boxes, gt_boxes):
    """
    Args:
        boxes: shape(N,M,4)
        gt_boxes: shape(N,M,4)
    """
    boxes_area = area(boxes)
    gt_boxes_area = area(gt_boxes)
    intersection_boxes = intersection(boxes, gt_boxes)
    area_intersection = area(intersection_boxes)
    return area_intersection / (boxes_area + gt_boxes_area - area_intersection)


def intersection(bbox, gt_boxes):
    """
    Args:
        bbox: shape(N,M,4)
        gt_boxes: shape(N,M,4)
    Returns:
        intersection: shape(N,M,4)
    """
    xmin = torch.max(gt_boxes[:, :, 0], bbox[:, :, 0])
    ymin = torch.max(gt_boxes[:, :, 1], bbox[:, :, 1])
    xmax = torch.min(gt_boxes[:, :, 2], bbox[:, :, 2])
    ymax = torch.min(gt_boxes[:, :, 3], bbox[:, :, 3])

    # if no intersection
    w = xmax - xmin + 1
    h = ymax - ymin + 1
    cond = (w < 0) | (h < 0)
    # xmin[cond] = 0
    # xmax[cond] = 0
    # ymin[cond] = 0
    # ymax[cond] = 0
    inter_boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
    inter_boxes[cond] = 0
    return inter_boxes


def super_nms(bboxes, nms_thresh=0.8, nms_num=2, loop_time=1):
    """
    Args:
        bboxes: shape(N,4)
    """
    similarity_calc_config = {"type": "center"}
    similarity_calc = similarity_calcs.build(similarity_calc_config)
    # expand batch dim
    bboxes = bboxes.unsqueeze(0)
    # shape(N,M,M)
    match_quality_matrix = similarity_calc.compare_batch(bboxes, bboxes)
    # squeeze back
    # shape(M,M)
    match_quality_matrix = match_quality_matrix[0]
    bboxes = bboxes[0]

    match_mask = match_quality_matrix > nms_thresh
    match_quality_matrix[match_mask] = 1
    match_quality_matrix[~match_mask] = 0

    for i in range(loop_time):
        # shape(M,)
        match_num = match_quality_matrix.sum(dim=-1)
        remain_unmatch_inds = torch.nonzero(match_num <= nms_num + 1)[:, 0]
        match_quality_matrix = match_quality_matrix.transpose(0, 1)
        match_quality_matrix[remain_unmatch_inds] = 0
        match_quality_matrix = match_quality_matrix.transpose(0, 1)

    match_num = match_quality_matrix.sum(dim=-1)
    # exclude self
    remain_match_inds = torch.nonzero(match_num > (nms_num + 1))
    if remain_match_inds.numel():
        remain_match_inds = remain_match_inds[:, 0]

    return remain_match_inds


def single_iou(box1, box2):
    """
    Args:
        box1: [x1,y1,x2,y2]
        box2: [x1,y1,x2,y2]
    """
    xmin = np.maximum(box1[0], box2[0])
    ymin = np.maximum(box1[1], box2[1])
    xmax = np.minimum(box1[2], box2[2])
    ymax = np.minimum(box1[3], box2[3])
    w = np.maximum(xmax - xmin, 0)
    h = np.maximum(ymax - ymin, 0)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return w * h / (area1 + area2)


def filter_by_center(boxes_centers, cluster_centers, min_dist=100):
    dist = torch.norm(boxes_centers - cluster_centers, dim=-1)
    return dist < min_dist


def super_nms_faster(boxes):
    """
    Args:
        boxes: shape(N, 4)
    Returns:
        keep
    """
    # min_iou = 0.8  # min iou
    # boxes_np = boxes.cpu().numpy()
    # import scipy.cluster.hierarchy as hcluster
    # clusters_np = hcluster.fclusterdata(boxes_np, min_iou, metric=single_iou)
    boxes_xy = geometry_utils.torch_xyxy_to_xywh(boxes.unsqueeze(0)).squeeze(0)
    xmin = boxes[:, ::2].min()
    xmax = boxes[:, ::2].max()
    ymin = boxes[:, 1::2].min()
    ymax = boxes[:, 1::2].max()

    x_slices = 10
    y_slices = 10
    x_stride = (xmax - xmin) / x_slices
    y_stride = (ymax - ymin) / y_slices
    cluster_x = torch.arange(0, x_slices) * x_stride
    cluster_y = torch.arange(0, y_slices) * y_stride
    xv, yv = torch.meshgrid([cluster_x, cluster_y])
    cluster = torch.stack(
        [xv.contiguous().view(-1),
         yv.contiguous().view(-1)], dim=-1).cuda().float()

    remain_boxes = []
    for i in range(cluster.shape[0]):
        mask = filter_by_center(boxes_xy[:, :2], cluster[i])
        cluster_boxes = boxes[mask]
        keep = super_nms(cluster_boxes, nms_thresh=0.8, nms_num=4, loop_time=1)
        if keep.numel() > 0:
            remain_boxes.append(keep)
    return torch.cat(remain_boxes, dim=0)
