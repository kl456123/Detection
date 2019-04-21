# -*- coding: utf-8 -*-

import logging
import time
from core.utils import common
import torch
import numpy as np

from lib.model.roi_layers import nms
from core import constants
import sys
import os
from utils import geometry_utils


class Tester(object):
    def __init__(self, eval_config, logger=None):
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger
        self.feat_vis = eval_config['feat_vis']
        self.thresh = eval_config['thresh']
        self.nms = eval_config['nms']
        self.class_agnostic = eval_config['class_agnostic']
        self.classes = ['bg'] + eval_config['classes']
        self.n_classes = len(self.classes)
        # self.batch_size = eval_config['batch_size']

        self.eval_out = eval_config['eval_out']

    def _generate_label_path(self, image_path):
        image_name = os.path.basename(image_path)
        sample_name = os.path.splitext(image_name)[0]
        label_name = sample_name + '.txt'
        return os.path.join(self.eval_out, label_name)

    def save_mono_3d_dets(self, dets, label_path):
        res_str = []
        kitti_template = '{} -1 -1 -10 {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.8f}'
        # kitti_template = '{} -1 -1 -10 {:.3f} {:.3f} {:.3f} {:.3f} -1 -1 -1 -1000 -1000 -1000 -10 {:.8f}'
        with open(label_path, 'w') as f:
            for cls_ind, dets_per_classes in enumerate(dets):
                for det in dets_per_classes:
                    xmin, ymin, xmax, ymax, cf, l, h, w, ry, x, y, z = det
                    res_str.append(
                        kitti_template.format(self.classes[cls_ind], xmin,
                                              ymin, xmax, ymax, h, w, l, x, y,
                                              z, ry, cf))
                    # xmin, ymin, xmax, ymax, cf, h, w, l, x, y, z, alpha = det
                    # res_str.append(
                # kitti_template.format(self.classes[cls_ind], xmin,
                # ymin, xmax, ymax, h, w, l, x, y,
                # z, alpha, cf))
            f.write('\n'.join(res_str))

    def save_dets(self, dets, label_path):
        res_str = []
        # kitti_template = '{} -1 -1 -10 {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.8f}'
        kitti_template = '{} -1 -1 -10 {:.3f} {:.3f} {:.3f} {:.3f} -1 -1 -1 -1000 -1000 -1000 -10 {:.8f}'
        with open(label_path, 'w') as f:
            for cls_ind, dets_per_classes in enumerate(dets):
                for det in dets_per_classes:
                    xmin, ymin, xmax, ymax, cf = det
                    res_str.append(
                        kitti_template.format(self.classes[cls_ind], xmin,
                                              ymin, xmax, ymax, cf))
                    # xmin, ymin, xmax, ymax, cf, h, w, l, x, y, z, alpha = det
                    # res_str.append(
                # kitti_template.format(self.classes[cls_ind], xmin,
                # ymin, xmax, ymax, h, w, l, x, y,
                # z, alpha, cf))
            f.write('\n'.join(res_str))

    def test(self, dataloader, model, logger):
        self.logger.info('Start testing')
        num_samples = len(dataloader)

        if self.feat_vis:
            # enable it before forward pass
            model.enable_feat_vis()

        for step, data in enumerate(dataloader):
            start_time = time.time()
            data = common.to_cuda(data)
            image_path = data[constants.KEY_IMAGE_PATH]

            with torch.no_grad():
                prediction = model(data)
            duration_time = time.time() - start_time

            if self.feat_vis:
                featmaps_dict = model.get_feat()
                from utils.visualizer import FeatVisualizer
                feat_visualizer = FeatVisualizer()
                feat_visualizer.visualize_maps(featmaps_dict)

            # initialize dets for each classes
            # dets = [[] for class_ind in range(self.n_classes)]
            dets = [[]]

            scores = prediction[constants.KEY_CLASSES]
            boxes_2d = prediction[constants.KEY_BOXES_2D]
            dims = prediction[constants.KEY_DIMS]
            orients = prediction[constants.KEY_ORIENTS_V2]
            p2 = data[constants.KEY_STEREO_CALIB_P2_ORIG]

            batch_size = scores.shape[0]
            scores = scores.view(-1, self.n_classes)
            new_scores = torch.zeros_like(scores)
            _, scores_argmax = scores.max(dim=-1)
            row = torch.arange(0, scores_argmax.numel()).type_as(scores_argmax)
            new_scores[row, scores_argmax] = scores[row, scores_argmax]
            scores = new_scores.view(batch_size, -1, self.n_classes)

            #  if step == 6:
            #  import ipdb
            #  ipdb.set_trace()

            for batch_ind in range(batch_size):
                boxes_2d_per_img = boxes_2d[batch_ind]
                scores_per_img = scores[batch_ind]
                dims_per_img = dims[batch_ind]
                orients_per_img = orients[batch_ind]
                p2_per_img = p2[batch_ind]
                for class_ind in range(1, self.n_classes):
                    # cls thresh
                    inds = torch.nonzero(
                        scores_per_img[:, class_ind] > self.thresh).view(-1)
                    threshed_scores_per_img = scores_per_img[inds, class_ind]
                    if inds.numel() > 0:
                        # if self.class_agnostic:
                        threshed_boxes_2d_per_img = boxes_2d_per_img[inds]
                        threshed_dims_per_img = dims_per_img[inds]
                        threshed_orients_per_img = orients_per_img[inds]
                        # else:
                        # threshed_boxes_2d_per_img = boxes_2d_per_img[
                        # inds, class_ind * 4:class_ind * 4 + 4]
                        # concat boxes and scores
                        threshed_dets_per_img = torch.cat([
                            threshed_boxes_2d_per_img,
                            threshed_scores_per_img.unsqueeze(-1),
                            threshed_dims_per_img,
                            threshed_orients_per_img.unsqueeze(-1)
                        ],
                                                          dim=-1)

                        # sort by scores
                        _, order = torch.sort(threshed_scores_per_img, 0, True)
                        threshed_dets_per_img = threshed_dets_per_img[order]

                        # nms
                        keep = nms(threshed_dets_per_img[:, :4],
                                   threshed_dets_per_img[:, 4],
                                   self.nms).view(-1).long()
                        nms_dets_per_img = threshed_dets_per_img[keep].detach(
                        ).cpu().numpy()

                        # calculate location
                        location = geometry_utils.calc_location(
                            nms_dets_per_img[:, 5:8], nms_dets_per_img[:, :5],
                            nms_dets_per_img[:, 8], p2_per_img.cpu().numpy())
                        nms_dets_per_img = np.concatenate(
                            [nms_dets_per_img, location], axis=-1)

                        dets.append(nms_dets_per_img)
                    else:
                        dets.append([])
                label_path = self._generate_label_path(image_path[batch_ind])
                self.save_mono_3d_dets(dets, label_path)
                sys.stdout.write('\r{}/{},duration: {}'.format(
                    step + 1, num_samples, duration_time))
                sys.stdout.flush()
