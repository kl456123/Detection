# -*- coding: utf-8 -*-

import time
from torch.autograd import Variable
from lib.model.rpn.bbox_transform import bbox_transform_inv
from lib.model.rpn.bbox_transform import clip_boxes
from lib.model.nms.nms_wrapper import nms
from utils.visualize import save_pkl, visualize_bbox
import numpy as np
import torch
import os
import sys


def to_cuda(target):
    if isinstance(target, list):
        return [to_cuda(e) for e in target]
    elif isinstance(target, dict):
        return {key: to_cuda(target[key]) for key in target}
    elif isinstance(target, torch.Tensor):
        return target.cuda()


def test(eval_config, data_loader, model):
    """
    Only one image in batch is supported
    """
    num_samples = len(data_loader)
    for i, data in enumerate(data_loader):
        img_file = data['img_name']
        start_time = time.time()
        pred_boxes, scores, rois, anchors = im_detect(
            model, to_cuda(data), eval_config, im_orig=data['img_orig'])
        duration_time = time.time() - start_time

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        rois = rois.squeeze()
        # anchors = anchors.squeeze()

        classes = eval_config['classes']
        thresh = eval_config['thresh']

        dets = []
        res_rois = []
        res_anchors = []
        # nms
        for j in range(1, len(classes)):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if eval_config['class_agnostic']:
                    cls_boxes = pred_boxes[inds, :]
                    rois_boxes = rois[inds, :]
                    anchors_boxes = anchors[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                rois_dets = torch.cat((rois_boxes, cls_scores.unsqueeze(1)), 1)
                anchors_dets = torch.cat(
                    (anchors_boxes, cls_scores.unsqueeze(1)), 1)

                cls_dets = cls_dets[order]
                rois_dets = rois_dets[order]
                anchors_dets = anchors_dets[order]

                keep = nms(cls_dets, eval_config['nms'])

                cls_dets = cls_dets[keep.view(-1).long()]
                rois_dets = rois_dets[keep.view(-1).long()]
                anchors = anchors_dets[keep.view(-1).long()]

                dets.append(cls_dets.detach().cpu().numpy())
                res_rois.append(rois_dets.detach().cpu().numpy())
                res_anchors.append(anchors.detach().cpu().numpy())
            else:
                dets.append([])
                res_rois.append([])
                res_anchors.append([])
        save_dets(dets[0], img_file[0], 'kitti', eval_config['eval_out'])
        save_dets(res_rois[0], img_file[0], 'kitti',
                  eval_config['eval_out_rois'])
        save_dets(res_anchors[0], img_file[0], 'kitti',
                  eval_config['eval_out_anchors'])

        sys.stdout.write(
            '\r{}/{},duration: {}'.format(i + 1, num_samples, duration_time))
        sys.stdout.flush()


def im_detect(model, data, eval_config, im_orig=None):
    # fake label
    # gt_boxes = torch.zeros((1, 1, 5))
    # num_boxes = torch.Tensor(1)
    # gt_boxes, num_boxes = __change_into_variable([gt_boxes, num_boxes])
    im_info = data['im_info']
    with torch.no_grad():
        prediction = model(data)
    cls_prob = prediction['rcnn_cls_probs']
    rois = prediction['rois_batch']
    bbox_pred = prediction['rcnn_bbox_preds']
    anchors = prediction['second_rpn_anchors'][0]
    # anchors = prediction['anchors'][0]
    # anchors = None

    scores = cls_prob
    im_scale = im_info[0][2]
    boxes = rois.data[:, :, 1:5]
    if prediction.get('rois_scores') is not None:
        rois_scores = prediction['rois_scores']
        boxes = torch.cat([boxes, rois_scores], dim=2)

    # visualize rois
    # import ipdb
    # ipdb.set_trace()
    if im_orig is not None and eval_config['rois_vis']:
        visualize_bbox(im_orig.numpy()[0], boxes.cpu().numpy()[0], save=True)
        # visualize_bbox(im_orig.numpy()[0], anchors[0].cpu().numpy()[:100], save=True)

    if eval_config['bbox_reg']:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if eval_config['bbox_normalize_targets_precomputed']:
            # Optionally normalize targets by a precomputed mean and stdev
            if eval_config['class_agnostic']:
                box_deltas = box_deltas.view(
                    -1, 4) * torch.FloatTensor(eval_config[
                        'bbox_normalize_stds']).cuda() + torch.FloatTensor(
                            eval_config['bbox_normalize_means']).cuda()
                box_deltas = box_deltas.view(eval_config['batch_size'], -1, 4)
            else:
                box_deltas = box_deltas.view(
                    -1, 4) * torch.FloatTensor(eval_config[
                        'bbox_normalize_stds']).cuda() + torch.FloatTensor(
                            eval_config['bbox_normalize_means']).cuda()
                box_deltas = box_deltas.view(eval_config['batch_size'], -1,
                                             4 * len(eval_config['classes']))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))
    pred_boxes /= im_scale
    return pred_boxes, scores, rois[:, :, 1:5], anchors


def save_dets(dets, label_info, data_format='kitti', output_dir=''):
    if data_format == 'kitti':
        save_dets_kitti(dets, label_info, output_dir)
    else:
        raise ValueError('data format is not ')


def save_dets_kitti(dets, label_info, output_dir):
    class_name = 'Car'
    label_idx = os.path.splitext(label_info)[0][-6:]
    label_file = label_idx + '.txt'
    label_path = os.path.join(output_dir, label_file)
    res_str = []
    kitti_template = '{} -1 -1 -10 {:.3f} {:.3f} {:.3f} {:.3f} -1 -1 -1 -1000 -1000 -1000 -10 {:.8f}'
    with open(label_path, 'w') as f:
        for det in dets:
            xmin, ymin, xmax, ymax, cf = det
            res_str.append(
                kitti_template.format(class_name, xmin, ymin, xmax, ymax, cf))
        f.write('\n'.join(res_str))


def save_bev_map(bev_map, label_info, cache_dir):
    label_idx = os.path.splitext(label_info)[0][-6:]
    label_file = label_idx + '.pkl'
    pkl_path = os.path.join(cache_dir, label_file)
    save_pkl(bev_map.numpy(), pkl_path)
