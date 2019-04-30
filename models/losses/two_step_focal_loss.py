# Descartes, basic object detection laboratory
# Support python2.7, python3, based on Pytorch 1.0
# Author: Yang Maoke (maokeyang@deepmotion.ai)
# Copyright (c) 2019-present, DeepMotion


import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, num_classes=21):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes

    @staticmethod
    def cross_entropy_loss(x, y):
        """Cross entropy loss w/o averaging across all samples.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) cross entroy loss, sized [N,].
        """
        pt = F.softmax(x, dim=1).gather(1, y.view(-1, 1))
        return -torch.log(pt)

    @staticmethod
    def hard_negative_mining(conf_loss, pos, ignore):
        """Return negative indices that is 3x the number as postive indices.
        Args:
          conf_loss: (tensor) cross entroy loss between conf_preds and conf_targets, sized [N*8732,].
          pos: (tensor) positive(matched) box indices, sized [N, 8732].
          ignore
        Return:
          (tensor) negative indices, sized [N, 8732].
        """
        with torch.no_grad():
            batch_size, num_boxes = pos.size()

            conf_loss = conf_loss.view(batch_size, -1)                  # [N,8732]
            conf_loss[pos] = 0                                          # set pos boxes = 0, the rest are neg conf_loss
            conf_loss[ignore] = 0

            _, idx = conf_loss.sort(1, descending=True)                 # sort by neg conf_loss
            _, rank = idx.sort(1)                                       # [N,8732]

            num_pos = pos.long().sum(1)                                 # [N,1]
            num_neg = torch.clamp(3 * num_pos, min=1, max=num_boxes-1)  # [N,1]
            neg = rank < num_neg.unsqueeze(1).expand_as(rank)           # [N,8732]

        return neg

    def focal_loss(self, x, y, alpha=0.5, gamma=2., size_average=False):
        """Focal loss

        Args:
            x(tensor): size [N, D]
            y(tensor): size [N, ]
            alpha(float)
            gamma(float)
            size_average
        Returns:
            (tensor): focal loss
        """
        with torch.no_grad():
            alpha_t = torch.ones(x.size()) * alpha
            alpha_t[:, 0] = 1 - alpha
            alpha_t = alpha_t.cuda().gather(1, y.view(-1, 1))
        pt = F.softmax(x, dim=1).gather(1, y.view(-1, 1))
        _loss = -alpha_t * torch.log(pt) * torch.pow((1 - pt), gamma)

        if size_average:
            return torch.mean(_loss)
        else:
            return torch.sum(_loss)

    def forward(self, loc1_preds, loc2_preds, loc_targets, conf_preds, conf_targets, os_pred, os_target, is_print=True):
        """Compute loss between (loc_preds, loc_targets) and (conf_preds, conf_targets).
        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, 8732, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, 8732, 4].
          conf_preds: (tensor) predicted class confidences, sized [batch_size, 8732, num_classes].
          conf_targets: (tensor) encoded target classes, sized [batch_size, 8732].
          is_print: whether print loss

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + CrossEntropyLoss(conf_preds, conf_targets).
          loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
          conf_loss = CrossEntropyLoss(pos_conf_preds, pos_conf_targets)
                    + CrossEntropyLoss(neg_conf_preds, neg_conf_targets)
        """
        # os_loss
        with torch.no_grad():
            pos = os_target > 0
            obj_ness = F.softmax(os_pred, dim=2)
            obj_ness = obj_ness[:, :, 1]
            os_pos = obj_ness > 0.4

        os_loss = self.focal_loss(os_pred[os_target > -1].view(-1, 2), os_target[os_target > -1], alpha=0.25) * 10

        pos_num = torch.sum(pos).float()
        if pos_num.item() > 0:
            os_loss = os_loss / pos_num
        else:
            os_loss /= 500.

        with torch.no_grad():
            pos = conf_targets > 0  # [N,8732], pos means the box matched.
            ignore = conf_targets < 0
            num_matched_boxes = torch.sum(pos).float()
            regression_num = torch.sum(pos).float()
            if num_matched_boxes.item() == 0:
                print("no matched boxes")
            pos_loc_targets = loc_targets[pos].view(-1, 4)  # [pos,4]

        if regression_num.item() > 0:
            loc_loss1 = F.smooth_l1_loss(loc1_preds[pos].view(-1, 4), pos_loc_targets, size_average=False)
            loc_loss = F.smooth_l1_loss(loc2_preds[pos].view(-1, 4), pos_loc_targets, size_average=False)
            loc_loss = loc_loss * 0.5 + loc_loss1 * 0.35
            loc_loss /= regression_num
        else:
            return torch.zeros_like(os_loss), os_loss, torch.zeros_like(os_loss)

        # conf_loss
        if torch.sum(os_pos) > torch.sum(pos) * 10:
            with torch.no_grad():
                conf_targets[ignore] = 0
                conf_loss = self.cross_entropy_loss(conf_preds.view(-1, self.num_classes), conf_targets.view(-1))
                neg = self.hard_negative_mining(conf_loss, pos, ignore)  # [N,8732]
                os_pos[neg == 0] = 0  # remove easy negative

        os_pos[ignore] = 0
        conf_loss = F.cross_entropy(conf_preds[pos + os_pos], conf_targets[pos + os_pos], size_average=True)

        if is_print:
            print('loc_loss: %f, cls_loss: %f, os_loss: %f' % (loc_loss.item(), conf_loss.item(), os_loss.item()))

        return loc_loss , os_loss , conf_loss

    @staticmethod
    def __heatmpas_nms(heatmaps, heatmap_thresh=0.5):
        maps_left = torch.zeros(heatmaps.size()).to('cuda')
        maps_right = torch.zeros(heatmaps.size()).to('cuda')
        maps_top = torch.zeros(heatmaps.size()).to('cuda')
        maps_bottom = torch.zeros(heatmaps.size()).to('cuda')

        maps_left[:, 1:, :] = heatmaps[:, :-1, :]
        maps_right[:, :-1, :] = heatmaps[:, 1:, :]
        maps_top[:, :, 1:] = heatmaps[:, :, :-1]
        maps_bottom[:, :, :-1] = heatmaps[:, :, 1:]

        peaks_binary = (heatmaps > heatmap_thresh) * \
            (heatmaps > maps_left) * (heatmaps > maps_right) * \
            (heatmaps > maps_top) * (heatmaps > maps_bottom)

        peaks = torch.nonzero(peaks_binary)

        return peaks

    @staticmethod
    def __generate_peakmap(heatmap, peaks, max_idx):
        peakmap = torch.zeros(heatmap.size()).long()
        for point in peaks:
            idx = max_idx[point[0]][point[1]][point[2]]
            peakmap[point[0], point[1], point[2], idx] = 1

        return peakmap.to('cuda')


if __name__ == "__main__":
    l = FocalLoss()
    l.test_cross_entropy_loss()
