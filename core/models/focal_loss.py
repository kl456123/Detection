#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: YangMaoke, DuanZhixiang({maokeyang, zhixiangduan}@deepmotion.ai)
# Focal loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def one_hot_embeding(labels, num_classes):
    """Embeding labels to one-hot form.

    Args:
        labels(LongTensor): class labels
        num_classes(int): number of classes
    Returns:
        encoded labels, sized[N, #classes]

    """

    y = torch.eye(num_classes)  # [D, D]
    return y[labels]  # [N, D]


class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=1, gamma=0):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def focal_loss(self, x, y):
        """Focal loss

        Args:
            x(tensor): size [N, D]
            y(tensor): size [N, ]
        Returns:
            (tensor): focal loss

        """

        t = one_hot_embeding(y.data.cpu(), self.num_classes)
        t = Variable(t).cuda()  # [N, 20]

        logit = F.softmax(x, dim=-1)
        logit = logit.clamp(1e-7, 1. - 1e-7)
        conf_loss_tmp = -1 * t.float() * torch.log(logit)
        conf_loss_tmp = self.alpha * conf_loss_tmp * (1 - logit)**self.gamma
        conf_loss = conf_loss_tmp.sum(dim=-1)
        return conf_loss

    def forward(self, cls_preds, cls_targets, ignored_label=-1,
                is_print=False):
        """
        Args:
            cls_preds: shape(N[N1,N2,...],num_classes)
            cls_targets: shape(N[N1,N2,...])
        """

        # shape(N*K,num_classes)
        masked_cls_preds = cls_preds.view(-1, self.num_classes)
        # shape(N*K,)
        cls_loss = self.focal_loss(masked_cls_preds, cls_targets)

        # ignore loss of ignroed_label
        ignored_mask = cls_targets == ignored_label
        cls_loss[ignored_mask] = 0

        # shape(N,K)
        cls_loss = cls_loss.view_as(cls_targets)

        if is_print:
            print(('cls_loss: %.3f' % (cls_loss.mean())))
        return cls_loss
