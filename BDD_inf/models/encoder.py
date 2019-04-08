# Encode target locations and labels.
import math
import torch
import numpy
import torch.nn.functional as F

#from lib.layers.nms import nms
from BDD_inf.models.detection import RetinaPriorBox


class DataEncoder:

    def __init__(self, cfg, anchor_type='ssd', infer_mode=False):
        """Compute default box sizes with scale and aspect transform."""
        if anchor_type == 'retina':
            self.default_boxes = RetinaPriorBox()(cfg)
        else:
            self.default_boxes = RetinaPriorBox()(cfg)

        if infer_mode:
            self.default_boxes = self.default_boxes.cuda()

    def encode(self, boxes, classes, threshold=0.5):
        '''Transform target bounding boxes and class labels to SSD boxes and classes. Match each object box
        to all the default boxes, pick the ones with the Jaccard-Index > 0.5:
            Jaccard(A,B) = AB / (A+B-AB)
        Args:
          boxes: (tensor) object bounding boxes (xmin,ymin,xmax,ymax) of a image, sized [#obj, 4].
          classes: (tensor) object class labels of a image, sized [#obj,].
          threshold: (float) Jaccard index threshold
        Returns:
          boxes: (tensor) bounding boxes, sized [#obj, 8732, 4].
          classes: (tensor) class labels, sized [8732,]
        '''
        default_boxes = self.default_boxes
        wh = default_boxes[:, 2:]
        default_boxes = torch.cat([default_boxes[:, :2] - default_boxes[:, 2:] / 2,
                                   default_boxes[:, :2] + default_boxes[:, 2:] / 2], 1)  # xmin, ymin, xmax, ymax
        iou = self.box_iou(boxes, default_boxes)  # [#obj,8732]

        max_iou, max_anchor = iou.max(1)
        iou, max_idx = iou.max(0)  # [1,8732]
        max_idx.squeeze_(0)        # [8732,]
        iou.squeeze_(0)            # [8732,]

        boxes = boxes[max_idx]  # [8732,4]
        variances = [0.1, 0.2]
        xymin = (boxes[:, :2] - default_boxes[:, :2]) / (variances[0] * wh)
        xymax = (boxes[:, 2:] - default_boxes[:, 2:]) / (variances[0] * wh)
        loc = torch.cat([xymin, xymax], 1)  # [8732,4]

        neg = (iou < 0.4)
        ignore = (iou < threshold)
        neg[max_anchor] = 0
        ignore[max_anchor] = 0
        conf = 1 + classes[max_idx]  # [8732,], background class = 0
        conf[ignore] = -1  # ignore[0.4, 0.5]
        conf[neg] = 0  # background

        os = torch.ones(iou.size()).long()
        os[ignore] = -1
        os[neg] = 0

        return loc, conf, os

    def decode(self, loc, conf, obj_ness, Nt=0.5):
        '''Transform predicted loc/conf back to real bbox locations and class labels.
        Args:
          loc: (tensor) predicted loc, sized [8732,4].
          conf: (tensor) predicted conf, sized [8732,21].
          Nt: float, threshold for NMS
        Returns:
          boxes: (tensor) bbox locations, sized [#obj, 4].
          labels: (tensor) class labels, sized [#obj,1].
        '''
        has_obj = False
        variances = [0.1, 0.2]
        default_boxes = torch.cat([self.default_boxes[:, :2] - self.default_boxes[:, 2:] / 2,
                                   self.default_boxes[:, :2] + self.default_boxes[:, 2:] / 2], 1)
        xymin = loc[:, :2] * variances[0] * self.default_boxes[:, 2:] + default_boxes[:, :2]
        xymax = loc[:, 2:] * variances[0] * self.default_boxes[:, 2:] + default_boxes[:, 2:]
        boxes = torch.cat([xymin, xymax], 1)  # [8732,4]

        max_conf, labels = conf.max(1)  # [8732,1]
        obj_ness = F.softmax(obj_ness, dim=1)[:, 1]
        ids = obj_ness > 0.4
        obj_ness *= max_conf
        ids[labels == 0] = 0
        if torch.sum(ids) > 1:
            has_obj = True
        else:
            return 0, 0, 0, has_obj

        boxes, labels, obj_ness = boxes[ids], labels[ids], obj_ness[ids]
        #keep = nms(boxes * 1000, obj_ness, Nt)
        keep = self.nms(boxes, obj_ness, Nt)

        return boxes[keep], labels[keep], obj_ness[keep], has_obj

    def decode_loc_max_nms(self, loc, conf, obj_ness, Nt=0.4, It=0.1):
        '''Transform predicted loc/conf back to real bbox locations and class labels.
        Args:
          loc: (tensor) predicted loc, sized [8732,4].
          conf: (tensor) predicted conf, sized [8732,21].
          Nt: float, threshold for NMS
        Returns:
          boxes: (tensor) bbox locations, sized [#obj, 4].
          labels: (tensor) class labels, sized [#obj,1].
        '''
        has_obj = False
        variances = [0.1, 0.2]
        default_boxes = torch.cat([self.default_boxes[:, :2] - self.default_boxes[:, 2:] / 2,
                                   self.default_boxes[:, :2] + self.default_boxes[:, 2:] / 2], 1)
        xymin = loc[:, :2] * variances[0] * self.default_boxes[:, 2:] + default_boxes[:, :2]
        xymax = loc[:, 2:] * variances[0] * self.default_boxes[:, 2:] + default_boxes[:, 2:]
        boxes = torch.cat([xymin, xymax], 1)  # [8732,4]

        obj_ness = F.softmax(obj_ness, dim=1)
        obj_ness = obj_ness[:, 1]

        idx_ = 0
        feature_width = [96, 192]
        peak_idx = torch.zeros(obj_ness.size()).long().to('cuda')
        for i in range(6):
            idx = feature_width[0] * feature_width[1] * 12
            heatmap = obj_ness[idx_: idx_ + idx]
            heatmap = heatmap.view(feature_width[0], feature_width[1], -1)
            heatmap_max, max_idx = heatmap.max(2)
            peaks = self.__heatmpas_nms(heatmap_max, heatmap_thresh=It)
            peaks = self.__generate_peakmap(heatmap, peaks, max_idx)
            peak_idx[idx_: idx_ + idx] = peaks.view(-1)

            idx_ += idx
            feature_width = [int(feature_width[0] / 2), int(feature_width[1] / 2)]

        max_conf, labels = conf.max(1)  # [8732,1]
        ids = obj_ness > Nt
        obj_ness *= max_conf

        _id = peak_idx > 0
        ids += _id
        ids[labels == 0] = 0
        if torch.sum(ids).float() > 0:
            has_obj = True
        else:
            return 0, 0, 0, has_obj

        boxes, scores, labels = self.fusing_boxes(boxes[ids], obj_ness[ids], labels[ids])
        keep = self.nms(boxes, scores, Nt)
        return boxes[keep], labels[keep], scores[keep], has_obj

    def __heatmpas_nms(self, heatmaps, heatmap_thresh=0.5):
        maps_left = torch.zeros(heatmaps.size()).to('cuda')
        maps_right = torch.zeros(heatmaps.size()).to('cuda')
        maps_top = torch.zeros(heatmaps.size()).to('cuda')
        maps_bottom = torch.zeros(heatmaps.size()).to('cuda')

        maps_left[1:, :] = heatmaps[:-1, :]
        maps_right[:-1, :] = heatmaps[1:, :]
        maps_top[:, 1:] = heatmaps[:, :-1]
        maps_bottom[:, :-1] = heatmaps[:, 1:]

        peaks_binary = (heatmaps > heatmap_thresh) * \
            (heatmaps > maps_left) * (heatmaps > maps_right) * \
            (heatmaps > maps_top) * (heatmaps > maps_bottom)

        peaks = torch.nonzero(peaks_binary)

        return peaks

    def __generate_peakmap(self, heatmap, peaks, max_idx):
        peakmap = torch.zeros(heatmap.size()).long()
        for point in peaks:
            idx = max_idx[point[0]][point[1]]
            peakmap[point[0], point[1], idx] = 1

        return peakmap.to('cuda')

    def nms(self, bboxes, scores, Nt=0.5, mode='union'):
        '''Non maximum suppression.

        Args:
          bboxes: (tensor) bounding boxes, sized [N,4].
          scores: (tensor) bbox scores, sized [N,].
          threshold: (float) overlap threshold.
          mode: (str) 'union' or 'min'.

        Returns:
          keep: (tensor) selected indices.

        Ref:
          https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
        '''
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        _, order = scores.sort(0, descending=True)

        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0]
                keep.append(i)

            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h

            if mode == 'union':
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
            elif mode == 'min':
                ovr = inter / areas[order[1:]].clamp(max=areas[i])
            else:
                raise TypeError('Unknown nms mode: %s.' % mode)

            ids = (ovr <= Nt).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]

        return torch.LongTensor(keep)

    @staticmethod
    def box_iou(box1, box2):
        '''Compute the intersection over union of two set of boxes.
        The default box order is (xmin, ymin, xmax, ymax).
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
          order: (str) box order, either 'xyxy' or 'xywh'.
        Return:
          (tensor) iou, sized [N,M].
        Reference:
          https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
        '''

        # N = box1.size(0)
        # M = box2.size(0)

        lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
        iou = inter / (area1[:, None] + area2 - inter)

        return iou

    @staticmethod
    def distance(box1, box2):
        N = box1.size(0)
        M = box2.size(0)

        center1 = (box1[:, :2] + box1[:, 2:]) / 2
        center2 = (box2[:, :2] + box2[:, 2:]) / 2

        center1 = center1.unsqueeze(1).expand(N, M, 2)
        center2 = center2.unsqueeze(0).expand(N, M, 2)
        diff = center1 - center2
        d = torch.sqrt(diff[:, :, 0]**2 + diff[:, :, 1]**2)

        return d

    def fusing_boxes(self, bboxes, scores, labels, Nt=0.9):
        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        confidence, order = scores.sort(0, descending=True)

        box_list = []
        score_list = []
        label_list = []

        while order.numel() > 0:
            i = order[0]
            if order.numel() == 1:
                box = bboxes[i, :].cpu().numpy()
                score = scores[i].cpu().numpy()
                lbl = labels[i].cpu().numpy()
                box_list.append(box)
                score_list.append(score)
                label_list.append(lbl)
                break

            xx1 = x1[order].clamp(min=x1[i])
            yy1 = y1[order].clamp(min=y1[i])
            xx2 = x2[order].clamp(max=x2[i])
            yy2 = y2[order].clamp(max=y2[i])

            w = (xx2-xx1).clamp(min=0)
            h = (yy2-yy1).clamp(min=0)
            inter = w * h
            _iou = inter / (areas[i] + areas[order] - inter)

            # fusing boxes
            fusing_idx = (_iou > Nt)
            fusing_idx = order[fusing_idx]
            fusing_boxs = bboxes[fusing_idx]
            fusing_lbls = labels[fusing_idx]
            fusing_scrs = scores[fusing_idx]

            fusing_box = torch.mean(fusing_boxs, dim=0).cpu().numpy()
            fusing_lbl, fusing_scr = self.__vote_class(fusing_lbls, fusing_scrs)

            box_list.append(fusing_box)
            score_list.append(fusing_scr)
            label_list.append(fusing_lbl)

            ids = (_iou <= Nt)
            if torch.sum(ids) == 0:
                break
            order = order[ids]

        boxes = torch.from_numpy(numpy.array(box_list, dtype=float))
        scores = torch.from_numpy(numpy.array(score_list, dtype=float))
        labels = torch.from_numpy(numpy.array(label_list, dtype=int))

        return boxes, scores, labels

    @staticmethod
    def __vote_class(labels, scores):
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()

        record = dict()
        keys = []
        for idx, lbl in enumerate(labels):
            if str(lbl) in record:
                record[str(lbl)]['max_score'] = max(record[str(lbl)]['max_score'], scores[idx])
            else:
                keys.append(str(lbl))
                record[str(lbl)] = {
                    'max_score': scores[idx],
                }

        if keys.__len__() == 1:
            label = int(keys[0])
            score = record[keys[0]]['max_score']
        else:
            max_score = 0
            max_idx = 0
            for idx, key in enumerate(keys):
                tmp_score = record[key]['max_score']
                if tmp_score > max_score:
                    max_score = tmp_score
                    max_idx = max_idx
            score = max_score
            label = keys[max_idx]

        return label, score


class GaussianFilter(object):

    def __init__(self, kernel_size, sigma, channels):
        self.filter = self.get_filter(kernel_size, sigma, channels)

    @staticmethod
    def get_filter(kernel_size, sigma, channels):
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                           torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2. * variance))
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

        gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                          padding=kernel_size // 2, groups=channels, bias=False)

        gaussian_filter.weight.data = gaussian_kernel
        gaussian_filter.weight.requires_grad = False

        return gaussian_filter.cuda()

    def __call__(self, heatmap):
        heatmap = self.filter.forward(heatmap)

        return heatmap
