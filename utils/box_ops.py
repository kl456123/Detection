# -*- coding: utf-8 -*-
import torch
from core.similarity_calc.center_similarity_calc import CenterSimilarityCalc
import math


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes = boxes.clone()
    boxes[boxes < 0] = 0

    batch_x = im_shape[:, 1] - 1
    batch_y = im_shape[:, 0] - 1

    size = boxes.size()
    boxes = boxes.view(-1, 4)

    boxes[:, 0][boxes[:, 0] > batch_x] = batch_x
    boxes[:, 1][boxes[:, 1] > batch_y] = batch_y
    boxes[:, 2][boxes[:, 2] > batch_x] = batch_x
    boxes[:, 3][boxes[:, 3] > batch_y] = batch_y

    boxes = boxes.view(size)

    return boxes


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
    similarity_calc = CenterSimilarityCalc()
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


def to_norm(boxes_3d, ortho_rotate=False):
    # boxes_3d = np.asarray(boxes_3d).reshape(-1, 7)

    anchors = torch.zeros_like(boxes_3d[:, :7])

    # Set x, y, z
    anchors[:, [0, 1, 2]] = boxes_3d[:, [0, 1, 2]]

    # Dimensions along x, y, z
    box_l = boxes_3d[:, [3]]
    box_w = boxes_3d[:, [4]]
    box_h = boxes_3d[:, [5]]
    box_ry = boxes_3d[:, [6]]

    # Rotate to nearest multiple of 90 degrees
    if ortho_rotate:
        half_pi = math.pi / 2
        box_ry = torch.round(box_ry / half_pi) * half_pi

    cos_ry = torch.abs(torch.cos(box_ry))
    sin_ry = torch.abs(torch.sin(box_ry))

    # dim_x, dim_y, dim_z
    anchors[:, [3]] = box_l * cos_ry + box_w * sin_ry
    anchors[:, [4]] = box_h
    anchors[:, [5]] = box_w * cos_ry + box_l * sin_ry

    return anchors


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep[:count], count
