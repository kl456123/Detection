# -*- coding: utf-8 -*-


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    boxes[boxes < 0] = 0

    batch_x = im_shape[:, 1] - 1
    batch_y = im_shape[:, 0] - 1

    boxes[:, :, 0][boxes[:, :, 0] > batch_x] = batch_x
    boxes[:, :, 1][boxes[:, :, 1] > batch_y] = batch_y
    boxes[:, :, 2][boxes[:, :, 2] > batch_x] = batch_x
    boxes[:, :, 3][boxes[:, :, 3] > batch_y] = batch_y

    return boxes


def filter_window(anchors, window, allowed_border=0):
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
