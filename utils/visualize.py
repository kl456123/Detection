# -*- coding: utf-8 -*-
"""
tools for visualize box for selecting the best anchors scale
and ratios
"""

import cv2
import numpy as np
from generate_anchors import generate_anchors


def expand_anchors(anchors, feat_size=(24, 79), feat_stride=16):
    # initialize some params
    num_anchors = anchors.shape[0]
    feat_height, feat_width = feat_size

    shift_x = np.arange(0, feat_width) * feat_stride
    shift_y = np.arange(0, feat_height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(),
                        shift_y.ravel())).transpose()
    A = num_anchors
    K = shifts.shape[0]

    anchors = anchors.reshape(1, A, 4) + shifts.reshape(K, 1, 4)
    anchors = anchors.reshape(K * A, 4)
    return anchors


def visualize_bbox(img, bboxes, gt_bboxes=[], size=None, save=False):
    """
    Args:
        bboxes: non-normalized(N,4)
        img: non-noramlized (H,W,C)(bgr)
    """

    print("img shape: ", img.shape)
    #################################
    # Image
    ################################

    # do resize first
    if size is not None:
        img = cv2.resize(img)

    # do something visualization according to the num of channels of images
    num_channles = img.shape[-1]

    # all image in imgs_batch should be 3-channels,3-dims
    imgs_batch = []
    h, w = img.shape[:2]
    if num_channles == 1 or num_channles > 3:
        # gray image
        for idx in range(num_channles):
            blob = np.zeros((h, w, 3))
            blob[:, :, 0] = img[:, :, idx]
            imgs_batch.append(blob)
    elif num_channles == 3:
        # color image
        imgs_batch.append(img)
    # make array contiguous for used by cv2
    imgs_batch = [
        np.ascontiguousarray(
            img, dtype=np.uint8) for img in imgs_batch
    ]

    #####################################
    # BOX
    #####################################
    if not isinstance(bboxes, np.ndarray):
        bboxes = np.asarray(bboxes)
    bboxes = bboxes.astype(np.int)

    # display
    for idx, img in enumerate(imgs_batch):
        for i, box in enumerate(bboxes):
            cv2.rectangle(
                img, (box[0], box[1]), (box[2], box[3]),
                color=(55, 255, 155),
                thickness=2)
        for i, box in enumerate(gt_bboxes):
            cv2.rectangle(
                img, (box[0], box[1]), (box[2], box[3]),
                color=(255, 255, 255),
                thickness=2)
        cv2.imshow('test', img)
        cv2.waitKey(0)

        if save:
            img_path = 'res_%d.jpg' % idx
            cv2.imwrite(img_path, img)


def shift_bbox(bbox, translation):
    """
    Args:
        translation:(x_delta,y_delta)
    """
    bbox[:, 0] += translation[0]
    bbox[:, 1] += translation[1]
    bbox[:, 2] += translation[0]
    bbox[:, 3] += translation[1]
    return bbox


def read_img(img_name):
    return cv2.imread(img_name)


def test():
    img_name = './img3.jpg'
    img = read_img(img_name)
    cv2.imshow('test', img)
    cv2.waitKey(0)


def analysis_boxes(boxes):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    print('h: ', h)
    print('w: ', w)


if __name__ == '__main__':
    # test()
    img_name = '000000.png'
    img = read_img(img_name)
    scales = np.array([2, 3, 4])
    ratios = np.array([0.5, 1, 2])
    anchors = generate_anchors(base_size=16, scales=scales, ratios=ratios)
    anchors = expand_anchors(anchors)
    print(anchors)
    # anchors = shift_bbox(anchors,translation=(200,200))
    analysis_boxes(anchors)
    # anchors = [[100,100,300,300]]
    visualize_bbox(img, anchors)
