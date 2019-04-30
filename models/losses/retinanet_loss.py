# -*- coding: utf-8 -*-

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.
    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.
    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]
    return y[labels]  # [N,D]


class FocalLoss(nn.Module):
    def __init__(self, num_classes=20):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes

    def focal_loss(self, x, y):
        '''Focal loss.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2

        t = one_hot_embedding(y.data.cpu(), self.num_classes)  # [N,21]
        t = Variable(t).cuda()  # [N,20]

        p = x.sigmoid()
        pt = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
        w = alpha * t + (1 - alpha) * (1 - t
                                       )  # w = alpha if t > 0 else 1-alpha
        w = w * (1 - pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

    def focal_loss_alt(self, x, y, num_classes=None):
        '''Focal loss alternative.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        if num_classes == None:
            num_classes = self.num_classes

        t = one_hot_embedding(y.data.cpu(), num_classes)
        t = Variable(t).cuda()

        xt = x * (2 * t - 1)  # xt = x if t > 0 else -x
        pt = (2 * xt + 1).sigmoid()

        w = alpha * t + (1 - alpha) * (1 - t)
        loss = -w * pt.log() / 2
        return loss.sum()

    def forward(self, loc1_preds, loc2_preds, loc_targets, cls_preds,
                cls_targets, os_preds, os_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.long().sum()

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc1_preds)  # [N,#anchors,4]
        masked_loc1_preds = loc1_preds[mask].view(-1, 4)  # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1, 4)  # [#pos,4]
        loc1_loss = F.smooth_l1_loss(
            masked_loc1_preds, masked_loc_targets, size_average=False)

        masked_loc2_preds = loc2_preds[mask].view(-1, 4)  # [#pos,4]
        loc2_loss = F.smooth_l1_loss(
            masked_loc2_preds, masked_loc_targets, size_average=False)

        loc_loss = loc2_loss * 0.5 + loc1_loss * 0.35

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes)
        cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])

        os_pos_neg = os_targets > -1  # exclude ignored anchors
        os_mask = os_pos_neg.unsqueeze(2).expand_as(os_preds)
        masked_os_preds = os_preds[os_mask].view(-1, 2)
        os_loss = self.focal_loss_alt(masked_os_preds, os_targets[os_pos_neg],
                                      2)

        #  print(
        #  'loc_loss: %.3f | cls_loss: %.3f' %
        #  (loc_loss.item() / num_pos, cls_loss.item() / num_pos),
        #  end=' | ')
        loss = (loc_loss + cls_loss + os_loss) / num_pos
        return loss
