from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr

from .generate_anchors import generate_anchors
from .bbox_transform import clip_boxes, bbox_overlaps_batch, bbox_transform_batch

from lib.model.utils.net_utils import _smooth_l1_loss

DEBUG = False

try:
    long  # Python 2
except NameError:
    long = int  # Python 3


class _AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """

    def __init__(self, layer_config):
        super(_AnchorTargetLayer, self).__init__()
        ratios = layer_config['anchor_ratios']
        scales = layer_config['anchor_scales']
        feat_stride = layer_config['feat_stride']

        self._feat_stride = feat_stride
        self._scales = scales
        self._num_anchors = len(ratios) * len(scales)

        self.rpn_clobber_positives = layer_config['rpn_clobber_positives']
        self.rpn_negative_overlap = layer_config['rpn_negative_overlap']
        self.rpn_positive_overlap = layer_config['rpn_positive_overlap']
        self.rpn_batch_size = layer_config['rpn_batch_size']
        self.rpn_fg_fraction = layer_config['rpn_fg_fraction']
        self.rpn_bbox_inside_weights = layer_config['rpn_bbox_inside_weights']
        self.rpn_positive_weight = layer_config['rpn_positive_weight']

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = 0  # default is 0

    def forward(self, input):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors

        # import ipdb
        # ipdb.set_trace()
        rpn_cls_probs = input[0]
        gt_boxes = input[1]
        im_info = input[2]
        all_anchors = input[3]
        target_assigner = input[4]
        sampler = input[5]
        rpn_cls_probs = rpn_cls_probs[:, self._num_anchors:, :, :]

        # map of shape (..., H, W)
        height, width = rpn_cls_probs.size(2), rpn_cls_probs.size(3)

        batch_size = gt_boxes.size(0)

        A = self._num_anchors

        total_anchors = all_anchors.shape[0]

        keep = (
            (all_anchors[:, 0] >= -self._allowed_border) &
            (all_anchors[:, 1] >= -self._allowed_border) &
            (all_anchors[:, 2] < long(im_info[0][1]) + self._allowed_border) &
            (all_anchors[:, 3] < long(im_info[0][0]) + self._allowed_border))

        inds_inside = torch.nonzero(keep).view(-1)
        rpn_cls_probs = rpn_cls_probs.permute(
            0, 2, 3, 1).contiguous().view(batch_size, total_anchors)
        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
        rpn_cls_probs = rpn_cls_probs.t()[inds_inside].t()

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)
        bbox_inside_weights = gt_boxes.new(batch_size,
                                           inds_inside.size(0)).zero_()
        bbox_outside_weights = gt_boxes.new(batch_size,
                                            inds_inside.size(0)).zero_()

        rpn_cls_targets, rpn_reg_targets, rpn_cls_weights, rpn_reg_weights = target_assigner.assign(
            anchors, gt_boxes[:, :, :4], gt_labels=None)

        ##################################
        # match
        ##################################
        overlaps = bbox_overlaps_batch(anchors, gt_boxes)

        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)

        labels = rpn_cls_targets
        labels[rpn_cls_weights == 0] = -1


        pos_indicator = labels == 1
        indicator = ~(labels == -1)
        cls_criterion = rpn_cls_probs

        batch_sampled_mask = sampler.subsample_batch(
            self.rpn_batch_size,
            pos_indicator,
            criterion=cls_criterion,
            indicator=indicator)

        labels.view(-1)[~batch_sampled_mask.view(-1)] = -1

        ##############################################

        bbox_targets = rpn_reg_targets

        # use a single value instead of 4 values for easy index.
        # bbox_inside_weights[labels == 1] = self.rpn_bbox_inside_weights[0]

        # if self.rpn_positive_weight < 0:
        num_examples = torch.sum(labels[0] >= 0)
        positive_weights = 1.0 / num_examples.item()
        negative_weights = 1.0 / num_examples.item()

        bbox_outside_weights[labels == 1] = positive_weights
        bbox_outside_weights[labels == 0] = negative_weights
        # import ipdb
        # ipdb.set_trace()
        bbox_inside_weights = rpn_reg_weights
        # rpn_reg_weights *=positive_weights
        # bbox_outside_weights = rpn_reg_weights
        # bbox_inside_weights = rpn_reg_weights

        labels = _unmap(
            labels, total_anchors, inds_inside, batch_size, fill=-1)
        bbox_targets = _unmap(
            bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
        bbox_inside_weights = _unmap(
            bbox_inside_weights,
            total_anchors,
            inds_inside,
            batch_size,
            fill=0)
        bbox_outside_weights = _unmap(
            bbox_outside_weights,
            total_anchors,
            inds_inside,
            batch_size,
            fill=0)

        outputs = []

        labels = labels.view(batch_size, height, width, A).permute(
            0, 3, 1, 2).contiguous()
        labels = labels.view(batch_size, 1, A * height, width)
        outputs.append(labels)

        bbox_targets = bbox_targets.view(batch_size, height, width, A *
                                         4).permute(0, 3, 1, 2).contiguous()
        outputs.append(bbox_targets)

        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(
            batch_size, anchors_count, 1).expand(batch_size, anchors_count, 4)

        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()

        outputs.append(bbox_inside_weights)

        bbox_outside_weights = bbox_outside_weights.view(
            batch_size, anchors_count, 1).expand(batch_size, anchors_count, 4)
        bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()
        outputs.append(bbox_outside_weights)

        return outputs

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count,
                           data.size(2)).fill_(fill).type_as(data)
        ret[:, inds, :] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])
