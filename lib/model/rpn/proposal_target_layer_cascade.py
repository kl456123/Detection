
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
from .bbox_transform import bbox_overlaps_batch, bbox_transform_batch


class _ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def __init__(self, layer_config):
        super(_ProposalTargetLayer, self).__init__()
        self._num_classes = layer_config['nclasses']
        self.bbox_normalize_means = torch.FloatTensor(
            layer_config['bbox_normalize_means'])
        self.bbox_normalize_stds = torch.FloatTensor(
            layer_config['bbox_normalize_stds'])
        self.bbox_inside_weights = torch.FloatTensor(
            layer_config['bbox_inside_weights'])
        self.batch_size = layer_config['batch_size']
        self.fg_fraction = layer_config['fg_fraction']
        self.bbox_normalize_targets_precomputed = layer_config[
            'bbox_normalize_targets_precomputed']
        self.fg_thresh = layer_config['fg_thresh']
        self.bg_thresh = layer_config['bg_thresh']
        self.bg_thresh_lo = layer_config['bg_thresh_lo']
        # self.fg_bbox_thresh = layer_config['fg_bbox_thresh']

        self.use_focal_loss = layer_config['use_focal_loss']

    def forward(self, all_rois, gt_boxes, num_boxes=None, rpn_cls_score=None):

        self.bbox_normalize_means = self.bbox_normalize_means.type_as(gt_boxes)
        self.bbox_normalize_stds = self.bbox_normalize_stds.type_as(gt_boxes)
        self.BBOX_INSIDE_WEIGHTS = self.bbox_inside_weights.type_as(gt_boxes)

        gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_()
        gt_boxes_append[:, :, 1:5] = gt_boxes[:, :, :4]
        # # combined gt_boxes with gt_ry
        # gt_boxes = torch.cat([gt_boxes,gt_ry],dim=1)

        # Include ground-truth boxes in the set of candidate rois
        all_rois = torch.cat([all_rois, gt_boxes_append], 1)

        num_images = 1
        if self.use_focal_loss:
            # no need to subsample
            rois_per_image = all_rois.size(1)
        else:
            rois_per_image = int(self.batch_size / num_images)

        fg_rois_per_image = int(np.round(self.fg_fraction * rois_per_image))
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image

        labels, rois, bbox_targets, bbox_inside_weights = self._sample_rois_pytorch(
            all_rois, gt_boxes, fg_rois_per_image, rois_per_image,
            self._num_classes)

        bbox_outside_weights = (bbox_inside_weights > 0).float()

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _get_bbox_regression_labels_pytorch(self, bbox_target_data,
                                            labels_batch, num_classes):
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tw, th, cos, sin)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
        bbox_target (ndarray): b x N x 4K blob of regression targets
        bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """
        batch_size = labels_batch.size(0)
        rois_per_image = labels_batch.size(1)
        clss = labels_batch
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image,
                                            4).zero_()
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()

        for b in range(batch_size):
            # assert clss[b].sum() > 0
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)
            for i in range(inds.numel()):
                ind = inds[i]
                bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]
                bbox_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS

        return bbox_targets, bbox_inside_weights

    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4
        assert gt_rois.size(2) == 4

        targets = bbox_transform_batch(ex_rois, gt_rois)

        if self.bbox_normalize_targets_precomputed:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - self.bbox_normalize_means.expand_as(targets))
                       / self.bbox_normalize_stds.expand_as(targets))

        return targets

    def _sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image,
                             rois_per_image, num_classes):
        """Generate a random sample of RoIs comprising foreground and background
        examples.
        """
        # overlaps: (rois x gt_boxes)

        overlaps = bbox_overlaps_batch(all_rois, gt_boxes)

        max_overlaps, gt_assignment = torch.max(overlaps, 2)
        focal_bbox_inside_weights = self.calculate_bbox_inside_weights(
            max_overlaps)
        # _,order = torch.sort(max_overlaps,descending=False)

        batch_size = overlaps.size(0)

        offset = torch.arange(0, batch_size) * gt_boxes.size(1)
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment

        labels = gt_boxes[:,:,4].contiguous().view(-1)[offset.view(-1)]\
            .view(batch_size, -1)

        labels_batch = labels.new(batch_size, rois_per_image).zero_()
        rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
        gt_rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()
        # used for subsample bbox
        # fake_label_batch = labels.new(batch_size, rois_per_image).zero_()
        # (idx,cos,sin)
        # gt_ry_batch = all_rois.new(batch_size, rois_per_image, 2).zero_()
        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        for i in range(batch_size):

            fg_inds = torch.nonzero(max_overlaps[i] >= self.fg_thresh).view(-1)

            # ohem for fg
            fg_overlaps = max_overlaps[i][fg_inds]
            _, fg_order = torch.sort(fg_overlaps, descending=False)

            fg_num_rois = fg_inds.numel()

            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
            if self.use_focal_loss:
                # ensure all are reserved
                bg_inds = torch.nonzero(
                    max_overlaps[i] < self.fg_thresh).view(-1)
            else:
                bg_inds = torch.nonzero((max_overlaps[i] < self.bg_thresh) & (
                    max_overlaps[i] >= self.bg_thresh_lo)).view(-1)
            focal_bbox_inside_weights[i][bg_inds] = 0

            # ohem for bg
            bg_overlaps = max_overlaps[i][bg_inds]
            _, bg_order = torch.sort(bg_overlaps, descending=True)

            bg_num_rois = bg_inds.numel()

            if not self.use_focal_loss:
                if fg_num_rois > 0 and bg_num_rois > 0:
                    # sampling fg
                    fg_rois_per_this_image = min(fg_rois_per_image,
                                                 fg_num_rois)

                    # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                    # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                    # use numpy instead.
                    rand_num = torch.randperm(fg_num_rois).long().cuda()
                    rand_num = torch.from_numpy(
                        np.random.permutation(fg_num_rois)).type_as(
                            gt_boxes).long()
                    fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

                    # sampling bg
                    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image

                    # Seems torch.rand has a bug, it will generate very large number and make an error.
                    # We use numpy rand instead.
                    rand_num = np.floor(
                        np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                    rand_num = torch.from_numpy(rand_num).type_as(
                        gt_boxes).long()
                    # if bg_rois_per_this_image <= bg_num_rois:
                    # bg_inds = bg_inds[bg_order[:bg_rois_per_this_image]]
                    # else:
                    # random samples and samples of the smallest iou
                    # rand_num = np.floor(
                    # np.random.rand(bg_rois_per_this_image) *
                    # (bg_rois_per_this_image - bg_num_rois))
                    # rand_num = torch.from_numpy(rand_num).type_as(
                    # gt_boxes).long()
                    # bg_inds = torch.cat(bg_inds, bg_inds[rand_num])
                    bg_inds = bg_inds[rand_num]

                elif fg_num_rois > 0 and bg_num_rois == 0:
                    # sampling fg
                    # if rois_per_image <= fg_num_rois:
                    # fg_inds = fg_inds[fg_order[:rois_per_image]]
                    # else:
                    rand_num = np.floor(
                        np.random.rand(rois_per_image) * fg_num_rois)
                    rand_num = torch.from_numpy(rand_num).type_as(
                        gt_boxes).long()
                    # fg_inds = torch.cat(fg_inds[rand_num], fg_inds)
                    fg_inds = fg_inds[rand_num]
                    fg_rois_per_this_image = rois_per_image
                    bg_rois_per_this_image = 0
                elif bg_num_rois > 0 and fg_num_rois == 0:
                    # sampling bg
                    #rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
                    # if rois_per_image <= bg_num_rois:
                    # bg_inds = bg_inds[bg_order[:rois_per_image]]
                    # else:
                    rand_num = np.floor(
                        np.random.rand(rois_per_image) * bg_num_rois)
                    rand_num = torch.from_numpy(rand_num).type_as(
                        gt_boxes).long()

                    # bg_inds = torch.cat(bg_inds[rand_num], bg_inds)
                    bg_inds = bg_inds[rand_num]
                    bg_rois_per_this_image = rois_per_image
                    fg_rois_per_this_image = 0
                else:
                    raise ValueError(
                        "bg_num_rois = 0 and fg_num_rois = 0, this should not happen!"
                    )
            else:
                fg_rois_per_this_image = fg_inds.numel()
                bg_rois_per_this_image = bg_inds.numel()

            # if focal loss used,no need to subsample

            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)

            # Select sampled values from various arrays:
            labels_batch[i].copy_(labels[i][keep_inds])

            # Clamp labels for the background RoIs to 0
            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0

            rois_batch[i] = all_rois[i][keep_inds]
            rois_batch[i, :, 0] = i
            # rois_batch[i,:,5:] = gt_ry[i][gt_assignment[i][keep_inds]]

            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]

        bbox_target_data = self._compute_targets_pytorch(
            rois_batch[:, :, 1:5], gt_rois_batch[:, :, :4])

        # shape(N,rois_per_image,4)
        bbox_targets, bbox_inside_weights = \
            self._get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch, num_classes)
        if self.use_focal_loss:
            bbox_inside_weights = focal_bbox_inside_weights

        return labels_batch, rois_batch, bbox_targets, bbox_inside_weights

    def sample_rois(self, max_overlaps):
        """
        Returns bbox_inside_weights
        """
        pass

    def calculate_bbox_inside_weights(self, max_overlaps):
        """
        max_overlaps: shape(N,inside_num_all_rois)
        bbox_inside_weights: shape(N,num_all_rois)
        """
        gamma = 2
        batch_size = max_overlaps.shape[0]
        bbox_inside_weights = torch.pow(1 - max_overlaps, gamma)
        # 4 degree
        bbox_inside_weights = bbox_inside_weights.repeat(1, 4).view(
            batch_size, 4, -1).transpose(1, 2).contiguous()

        return bbox_inside_weights

    def compute_rpn_ap(self, rpn_cls_score, rois_label):
        """
        Args:
            rpn_cls_score: shape(N,2*num_anchors,H,W)
            rois_label: shape(N,M)
        """
        pass
        # rpn_cls_

    def compute_rpn_ar(self):
        pass
