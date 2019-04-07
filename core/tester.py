# -*- coding: utf-8 -*-

import time
from torch.autograd import Variable
from lib.model.rpn.bbox_transform import bbox_transform_inv
from lib.model.rpn.bbox_transform import clip_boxes
from lib.model.nms.nms_wrapper import nms
from utils.visualize import save_pkl, visualize_bbox
from utils.postprocess import mono_3d_postprocess_angle, mono_3d_postprocess_bbox, mono_3d_postprocess_depth
from utils.kitti_util import proj_3dTo2d
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


def get_avod_predicted_boxes_3d_and_scores(
        final_pred_boxes_3d, final_pred_orientations, final_pred_softmax):
    # Calculate difference between box_3d and predicted angle
    ang_diff = final_pred_boxes_3d[:, 6] - final_pred_orientations

    # Wrap differences between -pi and pi
    two_pi = 2 * np.pi
    ang_diff[ang_diff < -np.pi] += two_pi
    ang_diff[ang_diff > np.pi] -= two_pi

    def swap_boxes_3d_lw(boxes_3d):
        boxes_3d_lengths = np.copy(boxes_3d[:, 3])
        boxes_3d[:, 3] = boxes_3d[:, 4]
        boxes_3d[:, 4] = boxes_3d_lengths
        return boxes_3d

    pi_0_25 = 0.25 * np.pi
    pi_0_50 = 0.50 * np.pi
    pi_0_75 = 0.75 * np.pi

    # Rotate 90 degrees if difference between pi/4 and 3/4 pi
    rot_pos_90_indices = np.logical_and(pi_0_25 < ang_diff, ang_diff < pi_0_75)
    final_pred_boxes_3d[rot_pos_90_indices] = \
        swap_boxes_3d_lw(final_pred_boxes_3d[rot_pos_90_indices])
    final_pred_boxes_3d[rot_pos_90_indices, 6] += pi_0_50

    # Rotate -90 degrees if difference between -pi/4 and -3/4 pi
    rot_neg_90_indices = np.logical_and(-pi_0_25 > ang_diff,
                                        ang_diff > -pi_0_75)
    final_pred_boxes_3d[rot_neg_90_indices] = \
        swap_boxes_3d_lw(final_pred_boxes_3d[rot_neg_90_indices])
    final_pred_boxes_3d[rot_neg_90_indices, 6] -= pi_0_50

    # Flip angles if abs difference if greater than or equal to 135
    # degrees
    swap_indices = np.abs(ang_diff) >= pi_0_75
    final_pred_boxes_3d[swap_indices, 6] += np.pi

    # Wrap to -pi, pi
    above_pi_indices = final_pred_boxes_3d[:, 6] > np.pi
    final_pred_boxes_3d[above_pi_indices, 6] -= two_pi

    # Find max class score index
    not_bkg_scores = final_pred_softmax[:, 1:]
    final_pred_types = np.argmax(not_bkg_scores, axis=1)

    # Take max class score (ignoring background)
    final_pred_scores = np.array([])
    for pred_idx in range(len(final_pred_boxes_3d)):
        all_class_scores = not_bkg_scores[pred_idx]
        max_class_score = all_class_scores[final_pred_types[pred_idx]]
        final_pred_scores = np.append(final_pred_scores, max_class_score)

    # Stack into prediction format
    predictions_and_scores = np.column_stack(
        [final_pred_boxes_3d, final_pred_scores])

    return predictions_and_scores


def test(eval_config, data_loader, model):
    num_samples = len(data_loader)

    for i, data in enumerate(data_loader):
        img_file = data['img_name']
        start_time = time.time()

        with torch.no_grad():
            data = to_cuda(data)
            prediction = model(data)

        duration_time = time.time() - start_time

        # import ipdb
        # ipdb.set_trace()
        pred_probs_3d = prediction['all_cls_softmax'].detach().cpu().numpy()
        final_pred_orientations = prediction['final_pred_orientations'].detach(
        ).cpu().numpy()
        final_pred_boxes_3d = prediction['final_pred_boxes_3d'].detach().cpu(
        ).numpy()
        predictions_and_scores = get_avod_predicted_boxes_3d_and_scores(
            final_pred_boxes_3d, final_pred_orientations, pred_probs_3d)

        p2 = data['stereo_calib_p2'][0].detach().cpu().numpy()
        cls_boxes = proj_3dTo2d(predictions_and_scores, p2)
        final_dets = np.concatenate(
            [cls_boxes, predictions_and_scores], axis=-1)

        thresh = eval_config['thresh']
        #  thresh = 0.0
        final_dets = final_dets[final_dets[:, -1] > thresh]

        save_dets(final_dets, img_file[0], 'kitti', eval_config['eval_out'])

        sys.stdout.write(
            '\r{}/{},duration: {}'.format(i + 1, num_samples, duration_time))
        sys.stdout.flush()


def _test(eval_config, data_loader, model):
    """
    Only one image in batch is supported
    """
    # import ipdb
    # ipdb.set_trace()
    num_samples = len(data_loader)
    for i, data in enumerate(data_loader):
        #  if i + 1 < 168:
        #  continue
        # else:
        # import ipdb
        # ipdb.set_trace()
        img_file = data['img_name']
        start_time = time.time()

        with torch.no_grad():
            data = to_cuda(data)
            prediction = model(data)

        import ipdb
        ipdb.set_trace()
        pred_probs_3d = prediction['all_cls_softmax']
        # pred_boxes_3d = prediction['final_bboxes_3d']
        final_pred_orientations = prediction['final_pred_orientations']
        final_pred_boxes_3d = prediction['final_pred_boxes_3d']
        final_pred_boxes_3d = get_avod_predicted_boxes_3d_and_scores(
            final_pred_boxes_3d.detach().cpu().numpy(),
            final_pred_orientations.detach().cpu().numpy())
        final_pred_boxes_3d = torch.tensor(final_pred_boxes_3d).type_as(
            pred_probs_3d)
        # pred_boxes_3d = data['anchor_boxes_3d_to_use']
        #  pred_boxes_3d = prediction['proposals_batch']
        #  pred_probs_3d = torch.ones_like(pred_boxes_3d[:, :2])

        duration_time = time.time() - start_time

        scores = pred_probs_3d
        pred_boxes_3d = final_pred_boxes_3d

        classes = eval_config['classes']
        thresh = eval_config['thresh']
        # import ipdb
        # ipdb.set_trace()
        # thresh = 0.0
        # import ipdb
        # ipdb.set_trace()

        dets = []
        # nms
        # import ipdb
        # ipdb.set_trace()
        for j in range(1, len(classes)):
            try:
                inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            except:
                import ipdb
                ipdb.set_trace()
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if eval_config['class_agnostic']:
                    pred_boxes_3d = pred_boxes_3d[inds, :]
                else:
                    pred_boxes_3d = pred_boxes_3d[inds][:, j * 4:(j + 1) * 4]

                pred_boxes_3d = pred_boxes_3d[order]

                # keep = nms(pred_boxes_3d, eval_config['nms'])

                # pred_boxes_3d = pred_boxes_3d[keep.view(-1).long()]

                pred_boxes_3d = pred_boxes_3d.detach().cpu().numpy()
                p2 = data['stereo_calib_p2'][0].detach().cpu().numpy()
                cls_scores = cls_scores.cpu().numpy()

                #  import ipdb
                #  ipdb.set_trace()
                cls_boxes = proj_3dTo2d(pred_boxes_3d, p2)
                #  cls_boxes = data['anchor_boxes_2d_norm'].detach().cpu().numpy()

                # import ipdb
                # ipdb.set_trace()
                cls_dets = np.concatenate(
                    (cls_boxes, cls_scores[..., np.newaxis]), 1)
                dets.append(np.concatenate([cls_dets, pred_boxes_3d], axis=-1))

            else:
                dets.append([])

        save_dets(dets[0], img_file[0], 'kitti', eval_config['eval_out'])

        sys.stdout.write(
            '\r{}/{},duration: {}'.format(i + 1, num_samples, duration_time))
        sys.stdout.flush()


def old_test(eval_config, data_loader, model):
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
                encoded_side_points = data['encoded_side_points'][0].detach(
                ).cpu().numpy()
                points_3d = points_3d.T

                p2 = data['p2'][0].detach().cpu().numpy()
                rcnn_3d = rcnn_3d.detach().cpu().numpy()
                # rcnn_3d_gt = rcnn_3d_gt.detach().cpu().numpy()

                # use gt
                use_gt = False

                if use_gt:
                    import ipdb
                    ipdb.set_trace()

                    center_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
                    center_y = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
                    gt_boxes_w = gt_boxes[:, 2] - gt_boxes[:, 0] + 1
                    gt_boxes_h = gt_boxes[:, 3] - gt_boxes[:, 1] + 1
                    center = np.stack([center_x, center_y], axis=-1)
                    gt_boxes_dims = np.stack([gt_boxes_w, gt_boxes_h], axis=-1)

                    point1 = encoded_side_points[:, :2] * gt_boxes_dims + center
                    point2 = encoded_side_points[:, 2:] * gt_boxes_dims + center

                    global_angles_gt = gt_boxes_3d[:, -1:]

                    rcnn_3d_gt = np.concatenate(
                        [gt_boxes_3d[:, :3], point1, point2], axis=-1)
                    # just for debug
                    if len(rcnn_3d_gt):
                        cls_dets_gt = np.concatenate(
                            [gt_boxes, np.zeros_like(gt_boxes[:, -1:])],
                            axis=-1)
                        rcnn_3d_gt, _ = mono_3d_postprocess_bbox(
                            rcnn_3d_gt, cls_dets_gt, p2)

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
                    rcnn_3d, location = mono_3d_postprocess_bbox(rcnn_3d,
                                                                 cls_dets, p2)
                    # rcnn_3d = mono_3d_postprocess_angle(rcnn_3d, cls_dets, p2)
                    # rcnn_3d = mono_3d_postprocess_depth(rcnn_3d, cls_dets, p2)
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
            # xmin, ymin, xmax, ymax, cf, h, w, l, x, y, z, alpha = det
            xmin, ymin, xmax, ymax, x, y, z, l, w, h, alpha, cf = det
            res_str.append(
                kitti_template.format(class_name, xmin, ymin, xmax, ymax, h, w,
                                      l, x, y, z, alpha, cf))
        f.write('\n'.join(res_str))


def save_bev_map(bev_map, label_info, cache_dir):
    label_idx = os.path.splitext(label_info)[0][-6:]
    label_file = label_idx + '.pkl'
    pkl_path = os.path.join(cache_dir, label_file)
    save_pkl(bev_map.numpy(), pkl_path)
