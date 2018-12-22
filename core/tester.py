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
from utils.box_ops import super_nms


def to_cuda(target):
    if isinstance(target, list):
        return [to_cuda(e) for e in target]
    elif isinstance(target, dict):
        return {key: to_cuda(target[key]) for key in target}
    elif isinstance(target, torch.Tensor):
        return target.cuda()


def _test(eval_config, data_loader, model):
    """
    Only one image in batch is supported
    """
    # seed = 0
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    num_samples = len(data_loader)
    # model.train()
    num_gt = 0
    matched = 0

    for i, data in enumerate(data_loader):
        img_file = data['img_name']

        start_time = time.time()
        pred_boxes, scores, rois, anchors, rois_scores = im_detect(
            model, to_cuda(data), eval_config, im_orig=data['img_orig'])
        duration_time = time.time() - start_time

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        rois = rois.squeeze()
        rois_scores = rois_scores.squeeze()

        thresh = eval_config['thresh']
        # import ipdb
        # ipdb.set_trace()
        use_which_result = eval_config['use_which_result']
        if not use_which_result == 'none':
            if 'rpn' in use_which_result:
                stats = model.stats
                match_inds = model.stats['match_inds']
            elif 'rcnn' in use_which_result:
                stats = model.rcnn_stats
            if 'un' in use_which_result:
                match_inds = stats['unmatch_inds']
            else:
                match_inds = stats['match_inds']

            thresh = 0
            eval_config['nms'] = 1
            scores = scores[match_inds]
            rois_scores = rois_scores[match_inds]
            pred_boxes = pred_boxes[match_inds]
            rois = rois[match_inds]
            anchors = anchors[match_inds]
        # anchors = anchors.squeeze()

        # calc recall
        matched += model.rcnn_stats['matched']
        num_gt += model.rcnn_stats['num_gt']
        rate = model.rcnn_stats['rate']
        fake_match_thresh = eval_config['fake_match_thresh']
        max_matched_ind = model.rcnn_stats['match_inds'].max()
        num_tp = model.rcnn_stats['num_tp']

        #  import ipdb
        #  ipdb.set_trace()
        # get remain tp after iou thresh
        # try:
        # remain_num_tp = torch.nonzero(
        # model.rcnn_stats['match'] > fake_match_thresh)[:, 1].numel()
        # except:
        remain_num_tp = 1
        test_ap = num_tp / remain_num_tp

        # try:
        # max_score = scores[torch.nonzero(model.rcnn_stats['iou'] < 0.3)
        # [:, 1]][:, 1].max()
        # except:
        max_score = 0
        # try:
        # min_score = scores[torch.nonzero(model.rcnn_stats['iou'] > 0.7)
        # [:, 1]][:, 1].min()
        # except:
        min_score = 0
        print("max_score(iou<0.3)/min_score(iou>0.7): {}/{}".format(max_score,
                                                                    min_score))

        classes = eval_config['classes']

        #  import ipdb
        #  ipdb.set_trace()
        dets = []
        res_rois = []
        res_anchors = []
        # nms
        for j in range(1, len(classes)):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                rois_cls_scores = rois_scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if eval_config['class_agnostic']:
                    cls_boxes = pred_boxes[inds, :]
                    rois_boxes = rois[inds, :]
                    anchors_boxes = anchors[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                rois_dets = torch.cat(
                    (rois_boxes, rois_cls_scores.unsqueeze(1)), 1)
                # the same as rois'
                anchors_dets = torch.cat(
                    (anchors_boxes, rois_cls_scores.unsqueeze(1)), 1)

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
        # import ipdb
        # ipdb.set_trace()
        save_dets(dets[0], img_file[0], 'kitti', eval_config['eval_out'])
        save_dets(res_rois[0], img_file[0], 'kitti',
                  eval_config['eval_out_rois'])
        save_dets(res_anchors[0], img_file[0], 'kitti',
                  eval_config['eval_out_anchors'])

        sys.stdout.write(
            '\r{}/{},duration: {}, iou_rate/iou/max_ind: {}/{}/{}, num_tp/remain_num_tp/test_ap: {}/{}/{}'.
            format(i + 1, num_samples, duration_time, rate, fake_match_thresh,
                   max_matched_ind, num_tp, remain_num_tp, test_ap))
        sys.stdout.flush()
    print('\naverage recall/num_gt/matched: {:.4f}/{}/{}'.format(
        matched / num_gt, num_gt, matched))


def test(eval_config, data_loader, model):
    """
    Only one image in batch is supported
    """
    num_samples = len(data_loader)
    num_gt = 0
    matched = 0

    for i, data in enumerate(data_loader):
        img_file = data['img_name']

        start_time = time.time()
        pred_boxes, scores, rois, anchors, rois_scores = im_detect(
            model, to_cuda(data), eval_config, im_orig=data['img_orig'])
        duration_time = time.time() - start_time

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        rois = rois.squeeze()
        rois_scores = rois_scores.squeeze()

        # thresh = eval_config['thresh']
        use_which_result = eval_config['use_which_result']
        if not use_which_result == 'none':
            if 'rpn' in use_which_result:
                stats = model.stats
                match_inds = model.stats['match_inds']
            elif 'rcnn' in use_which_result:
                stats = model.rcnn_stats
            if 'un' in use_which_result:
                match_inds = stats['unmatch_inds']
            else:
                match_inds = stats['match_inds']

            eval_config['nms'] = 1
            scores = scores[match_inds]
            rois_scores = rois_scores[match_inds]
            pred_boxes = pred_boxes[match_inds]
            rois = rois[match_inds]
            anchors = anchors[match_inds]
        else:
            # import ipdb
            # ipdb.set_trace()
            # new postprocess
            keep = super_nms(pred_boxes, nms_num=3, nms_thresh=0.8, loop_time=10)
            # keep = torch.ones_like(keep)
            # print('num of keep {}'.format(keep.numel()))


            scores = scores[keep]
            rois_scores = rois_scores[keep]
            pred_boxes = pred_boxes[keep]
            rois = rois[keep]
            anchors = anchors[keep]

        # calc recall
        matched += model.rcnn_stats['matched']
        num_gt += model.rcnn_stats['num_gt']
        rate = model.rcnn_stats['rate']
        fake_match_thresh = eval_config['fake_match_thresh']
        max_matched_ind = model.rcnn_stats['match_inds'].max()
        num_tp = model.rcnn_stats['num_tp']

        remain_num_tp = 1
        test_ap = num_tp / remain_num_tp

        # max_score = 0
        # min_score = 0
        # print("max_score(iou<0.3)/min_score(iou>0.7): {}/{}".format(max_score,
        # min_score))

        classes = eval_config['classes']

        dets = []
        res_rois = []
        res_anchors = []
        # nms
        for j in range(1, len(classes)):
            #  inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if scores.numel() > 0:
                cls_scores = scores[:, j]
                rois_cls_scores = rois_scores[:, j]
                _, order = torch.sort(cls_scores, 0, True)
                if eval_config['class_agnostic']:
                    cls_boxes = pred_boxes
                    rois_boxes = rois
                    anchors_boxes = anchors
                else:
                    cls_boxes = pred_boxes[:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                rois_dets = torch.cat(
                    (rois_boxes, rois_cls_scores.unsqueeze(1)), 1)
                # the same as rois'
                anchors_dets = torch.cat(
                    (anchors_boxes, rois_cls_scores.unsqueeze(1)), 1)

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
            '\r{}/{},duration: {}, iou_rate/iou/max_ind: {}/{}/{}, num_tp/remain_num_tp/test_ap: {}/{}/{}'.
            format(i + 1, num_samples, duration_time, rate, fake_match_thresh,
                   max_matched_ind, num_tp, remain_num_tp, test_ap))
        # sys.stdout.write(
        # '\r{}/{},duration: {}'.format(i + 1, num_samples, duration_time))
        sys.stdout.flush()
    print('\naverage recall/num_gt/matched: {:.4f}/{}/{}'.format(
        matched / num_gt, num_gt, matched))


def im_detect(model, data, eval_config, im_orig=None):
    im_info = data['im_info']
    if eval_config.get('feat_vis'):
        # enable it before forward pass
        model.enable_feat_vis()

    with torch.no_grad():
        prediction = model(data)

    if eval_config.get('feat_vis'):
        featmaps_dict = model.get_feat()
        from utils.visualizer import FeatVisualizer
        feat_visualizer = FeatVisualizer()
        feat_visualizer.visualize_maps(featmaps_dict)

    second_rpn_cls_probs = prediction['second_rpn_cls_probs']
    if prediction.get('rcnn_cls_probs') is not None:
        cls_prob = prediction['rcnn_cls_probs']
    else:
        cls_prob = second_rpn_cls_probs
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
    #  import ipdb
    #  ipdb.set_trace()
    if im_orig is not None and eval_config['rois_vis']:
        visualize_bbox(im_orig.numpy()[0], boxes.cpu().numpy()[0], save=True)
        # visualize_bbox(im_orig.numpy()[0], anchors[0].cpu().numpy()[:100], save=True)

    if eval_config['bbox_reg']:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        pred_boxes = model.target_assigner.bbox_coder.decode_batch(
            box_deltas.view(eval_config['batch_size'], -1, 4), boxes)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

    pred_boxes /= im_scale
    rois /= im_scale
    anchors /= im_scale
    return pred_boxes, scores, rois[:, :, 1:5], anchors, second_rpn_cls_probs


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
        # if len(res_str) == 0:
        # print('empty file')
        f.write('\n'.join(res_str))
        f.flush()
        os.fsync(f)


def save_bev_map(bev_map, label_info, cache_dir):
    label_idx = os.path.splitext(label_info)[0][-6:]
    label_file = label_idx + '.pkl'
    pkl_path = os.path.join(cache_dir, label_file)
    save_pkl(bev_map.numpy(), pkl_path)
