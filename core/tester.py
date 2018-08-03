# -*- coding: utf-8 -*-

import time
from torch.autograd import Variable
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from utils.visualize import save_pkl, visualize_bbox
import numpy as np
import torch
import os
import sys


def __change_into_variable(elems, use_gpu=True, training=True):
    if not training:
        volatile = True
    else:
        volatile = False
    if use_gpu:
        return [Variable(elem.cuda(), volatile) for elem in elems]
    else:
        return [Variable(elem, volatile) for elem in elems]


def test(eval_config, data_loader, model):
    """
    Only one image in batch is supported
    """
    num_samples = len(data_loader)
    for i, data in enumerate(data_loader):
        im_data = data['img']
        im_info = data['im_info']
        img_file = data['img_name']
        # if eval_config['cache_bev']:
        # img_orig = data['img_orig']
        # save_bev_map(img_orig[0], img_file[0], eval_config['cache_dir'])
        # gt_boxes = data['bbox']
        # num_boxes = data['num']

        im_data, im_info = __change_into_variable([im_data, im_info])
        # det_tic = time.time()
        start_time = time.time()
        pred_boxes, scores = im_detect(
            model,
            im_data,
            img_file,
            im_info,
            eval_config,
            im_orig=data['img_orig'])
        duration_time = time.time() - start_time

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        # det_toc = time.time()
        # detect_time = det_toc - det_tic
        # misc_tic = time.time()
        classes = eval_config['classes']
        thresh = eval_config['thresh']

        dets = []
        # nms
        for j in range(1, len(classes)):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if eval_config['class_agnostic']:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, eval_config['nms'])
                cls_dets = cls_dets[keep.view(-1).long()]
                # if vis:
                # im2show = vis_detections(im2show, imdb.classes[j],
                # cls_dets.cpu().numpy(), 0.3)
                dets.append(cls_dets.cpu().numpy())
            else:
                dets.append([])
        save_dets(dets[0], img_file[0], 'kitti', eval_config['eval_out'])
        sys.stdout.write(
            '\r{}/{},duration: {}'.format(i + 1, num_samples, duration_time))
        sys.stdout.flush()


def im_detect(model, im_data, im_name, im_info, eval_config, im_orig=None):
    # fake label
    gt_boxes = torch.zeros((1, 1, 5))
    num_boxes = torch.Tensor(1)
    gt_boxes, num_boxes = __change_into_variable([gt_boxes, num_boxes])
    prediction = model(im_data, im_info, gt_boxes, num_boxes)
    cls_prob = prediction['cls_prob']
    rois = prediction['rois']
    bbox_pred = prediction['bbox_pred']

    scores = cls_prob.data
    im_scale = im_info.data[0][2]
    boxes = rois.data[:, :, 1:5]
    if prediction.get('rois_scores') is not None:
        rois_scores = prediction['rois_scores']
        boxes = torch.cat([boxes, rois_scores], dim=2)

    # visualize rois
    # import ipdb
    # ipdb.set_trace()
    rois = prediction['rois']
    if im_orig is not None and eval_config['rois_vis']:
        visualize_bbox(im_orig.numpy()[0], boxes.cpu().numpy()[0], save=True)

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
    return pred_boxes, scores


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
