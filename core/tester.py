# -*- coding: utf-8 -*-

import time
from torch.autograd import Variable
from lib.model.rpn.bbox_transform import bbox_transform_inv
from lib.model.rpn.bbox_transform import clip_boxes
from lib.model.nms.nms_wrapper import nms
from utils.visualize import save_pkl, visualize_bbox
from utils.postprocess import mono_3d_postprocess_angle, mono_3d_postprocess_bbox, mono_3d_postprocess_depth
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
        pred_boxes, scores, rois, anchors, rcnn_3d = im_detect(
            model, to_cuda(data), eval_config, im_orig=data['img_orig'])
        duration_time = time.time() - start_time

        # import ipdb
        # ipdb.set_trace()
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        rois = rois.squeeze()
        rcnn_3d = rcnn_3d.squeeze()
        # anchors = anchors.squeeze()

        classes = eval_config['classes']
        thresh = eval_config['thresh']

        # import ipdb
        # ipdb.set_trace()
        dets = []
        res_rois = []
        res_anchors = []
        dets_3d = []
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
                rcnn_3d = rcnn_3d[order]

                keep = nms(cls_dets, eval_config['nms'])

                cls_dets = cls_dets[keep.view(-1).long()]
                rois_dets = rois_dets[keep.view(-1).long()]
                anchors = anchors_dets[keep.view(-1).long()]
                rcnn_3d = rcnn_3d[keep.view(-1).long()]

                cls_dets = cls_dets.detach().cpu().numpy()
                res_rois.append(rois_dets.detach().cpu().numpy())
                res_anchors.append(anchors.detach().cpu().numpy())

                coords = data['coords'][0].detach().cpu().numpy()
                gt_boxes = data['gt_boxes'][0].detach().cpu().numpy()
                gt_boxes_3d = data['gt_boxes_3d'][0].detach().cpu().numpy()
                points_3d = data['points_3d'][0].detach().cpu().numpy()
                local_angles_gt = data['local_angle'][0].detach().cpu().numpy()
                local_angle_oritation_gt = data['local_angle_oritation'][
                    0].detach().cpu().numpy()
                points_3d = points_3d.T

                p2 = data['p2'][0].detach().cpu().numpy()
                rcnn_3d = rcnn_3d.detach().cpu().numpy()
                # rcnn_3d_gt = rcnn_3d_gt.detach().cpu().numpy()

                # use gt
                use_gt = False

                if use_gt:
                    import ipdb
                    ipdb.set_trace()

                    global_angles_gt = gt_boxes_3d[:, -1:]
                    rcnn_3d_gt = np.concatenate(
                        [gt_boxes_3d[:, :3], global_angles_gt], axis=-1)
                    # just for debug
                    if len(rcnn_3d_gt):
                        cls_dets_gt = np.concatenate(
                            [gt_boxes, np.zeros_like(gt_boxes[:, -1:])],
                            axis=-1)
                        rcnn_3d_gt = mono_3d_postprocess_bbox(rcnn_3d_gt,
                                                              cls_dets_gt, p2)

                        dets.append(
                            np.concatenate(
                                [cls_dets_gt, rcnn_3d_gt], axis=-1))
                    else:
                        dets.append([])
                        res_rois.append([])
                        res_anchors.append([])
                        dets_3d.append([])
                else:
                    # import ipdb
                    # ipdb.set_trace()
                    # sample_name = os.path.splitext(os.path.basename(data['img_name'][0]))[0]
                    # if sample_name=='000031':
                    # import ipdb
                    # ipdb.set_trace()
                    #  rcnn_3d[:, :-1] = gt_boxes_3d[:, :3]
                    # global_angles_gt = gt_boxes_3d[:, -1:]
                    # rcnn_3d = np.concatenate(
                    # [gt_boxes_3d[:, :3], global_angles_gt], axis=-1)
                    # rcnn_3d[:,3] = 1-rcnn_3d[:,3]
                    # import ipdb
                    # ipdb.set_trace()
                    # rcnn_3d, location = mono_3d_postprocess_bbox(rcnn_3d, cls_dets, p2)
                    # rcnn_3d = mono_3d_postprocess_angle(rcnn_3d, cls_dets, p2)
                    rcnn_3d = mono_3d_postprocess_depth(rcnn_3d, cls_dets, p2)
                    # rcnn_3d[:, 3:6] = location
                    # rcnn_3d = np.zeros((cls_dets.shape[0], 7))
                    dets.append(np.concatenate([cls_dets, rcnn_3d], axis=-1))

            else:
                dets.append([])
                res_rois.append([])
                res_anchors.append([])
                dets_3d.append([])

        # import ipdb
        # ipdb.set_trace()
        save_dets(dets[0], img_file[0], 'kitti', eval_config['eval_out'])
        # save_dets(res_rois[0], img_file[0], 'kitti',
        # eval_config['eval_out_rois'])
        # save_dets(res_anchors[0], img_file[0], 'kitti',
        # eval_config['eval_out_anchors'])

        sys.stdout.write(
            '\r{}/{},duration: {}'.format(i + 1, num_samples, duration_time))
        sys.stdout.flush()


def decode_3d(rcnn_3ds, boxes_2d):
    """
    Args:
        rcnn_3ds: shape(N,7)
    """
    center_x = (boxes_2d[:, 2] + boxes_2d[:, 0]) / 2
    center_y = (boxes_2d[:, 3] + boxes_2d[:, 1]) / 2
    center = np.expand_dims(np.stack([center_x, center_y], axis=-1), axis=1)
    w = (boxes_2d[:, 2] - boxes_2d[:, 0] + 1)
    h = (boxes_2d[:, 3] - boxes_2d[:, 1] + 1)
    dims = np.expand_dims(np.stack([w, h], axis=-1), axis=1)
    rcnn_coords = rcnn_3ds[:, :-1].reshape((-1, 3, 2))
    rcnn_coords = rcnn_coords * dims + center

    y = rcnn_3ds[:, -1:] * dims[:, 0, 1:] + center[:, 0, 1:]
    return np.concatenate([rcnn_coords.reshape((-1, 6)), y], axis=-1)


def im_detect(model, data, eval_config, im_orig=None):
    # fake label
    # gt_boxes = torch.zeros((1, 1, 5))
    # num_boxes = torch.Tensor(1)
    # gt_boxes, num_boxes = __change_into_variable([gt_boxes, num_boxes])
    im_info = data['im_info']
    with torch.no_grad():
        prediction = model(data)

    if eval_config.get('feat_vis'):
        featmaps_dict = model.get_feat()
        from utils.visualizer import FeatVisualizer
        feat_visualizer = FeatVisualizer()
        feat_visualizer.visualize_maps(featmaps_dict)

    cls_prob = prediction['rcnn_cls_probs']
    rois = prediction['rois_batch']
    bbox_pred = prediction['rcnn_bbox_preds']
    anchors = prediction['second_rpn_anchors'][0]
    rcnn_3d = prediction['rcnn_3d']
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
        #  if eval_config['bbox_normalize_targets_precomputed']:
        #  # Optionally normalize targets by a precomputed mean and stdev
        #  if eval_config['class_agnostic']:
        #  box_deltas = box_deltas.view(
        #  -1, 4) * torch.FloatTensor(eval_config[
        #  'bbox_normalize_stds']).cuda() + torch.FloatTensor(
        #  eval_config['bbox_normalize_means']).cuda()
        #  box_deltas = box_deltas.view(eval_config['batch_size'], -1, 4)
        #  else:
        #  box_deltas = box_deltas.view(
        #  -1, 4) * torch.FloatTensor(eval_config[
        #  'bbox_normalize_stds']).cuda() + torch.FloatTensor(
        #  eval_config['bbox_normalize_means']).cuda()
        #  box_deltas = box_deltas.view(eval_config['batch_size'], -1,
        #  4 * len(eval_config['classes']))

        #  pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = model.target_assigner.bbox_coder.decode_batch(
            box_deltas.view(eval_config['batch_size'], -1, 4), boxes)
        # pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

    pred_boxes /= im_scale
    return pred_boxes, scores, rois[:, :, 1:5], anchors, rcnn_3d


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
    kitti_template = '{} -1 -1 -10 {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.8f}'
    with open(label_path, 'w') as f:
        for det in dets:
            xmin, ymin, xmax, ymax, cf, h, w, l, x, y, z, alpha = det
            res_str.append(
                kitti_template.format(class_name, xmin, ymin, xmax, ymax, h, w,
                                      l, x, y, z, alpha, cf))
        f.write('\n'.join(res_str))


def save_bev_map(bev_map, label_info, cache_dir):
    label_idx = os.path.splitext(label_info)[0][-6:]
    label_file = label_idx + '.pkl'
    pkl_path = os.path.join(cache_dir, label_file)
    save_pkl(bev_map.numpy(), pkl_path)
