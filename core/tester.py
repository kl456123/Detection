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
from utils import box_ops
from utils.drawer import ImageVisualizer
from utils import pointcloud_utils

# from tmp.utils.postprocess import mono_3d_postprocess_bbox


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
        self.test_type = eval_config['test_type']

        # image visualizer for any dataset
        image_dir = '/data/object/training/image_2'
        result_dir = './results/data'
        save_dir = 'results/images'
        calib_dir = '/data/object/training/calib'
        label_dir = None
        calib_file = None
        self.visualizer = ImageVisualizer(
            image_dir,
            result_dir,
            label_dir=label_dir,
            calib_dir=calib_dir,
            calib_file=calib_file,
            online=False,
            save_dir=save_dir)

    def _generate_label_path(self, image_path, eval_out=None):
        if eval_out is None:
            eval_out = self.eval_out
        image_name = os.path.basename(image_path)
        sample_name = os.path.splitext(image_name)[0]
        label_name = sample_name + '.txt'
        return os.path.join(eval_out, label_name)

    def save_pseudo_3d(self, dets_2d, corners_2d, dims, depth, label_path):
        """
        format: x1,y1,x2,y2,corners_2d, dims, depth
        """
        num_instances = dets_2d.shape[0]
        if num_instances == 0:
            return
        corners_2d = corners_2d.reshape(num_instances, -1)

        # dims = np.zeros((num_instances, 3))
        # depth = np.zeros((num_instances, 1))
        dets = np.concatenate(
            [dets_2d[:, :4], corners_2d, dims, depth, dets_2d[:, -1:]],
            axis=-1).astype(np.str)

        lines = []
        for det in dets:
            lines.append(' '.join(det))
        res_str = '\n'.join(lines)

        with open(label_path, 'w') as f:
            f.write(res_str)

    def save_mono_3d_dets(self, dets, label_path):
        res_str = []
        kitti_template = '{} -1 -1 -10 {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.8f}'
        # kitti_template = '{} -1 -1 -10 {:.3f} {:.3f} {:.3f} {:.3f} -1 -1 -1 -1000 -1000 -1000 -10 {:.8f}'
        with open(label_path, 'w') as f:
            for cls_ind, dets_per_classes in enumerate(dets):
                if self.classes[cls_ind] == 'Tram':
                    continue
                for det in dets_per_classes:
                    # xmin, ymin, xmax, ymax, cf, l, h, w, ry, x, y, z = det
                    xmin, ymin, xmax, ymax, cf, h, w, l, x, y, z, ry = det
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

    def save_dets(self, dets, label_path, image_path):
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

        # image = self.visualizer.parse_image(image_path)
        # self.visualizer.render_image_2d(image, boxes_2d, label_classes)

    def test_corners_3d(self, dataloader, model, logger):
        self.logger.info('Start testing')
        num_samples = len(dataloader)

        if self.feat_vis:
            # enable it before forward pass
            model.enable_feat_vis()
        end_time = 0

        for step, data in enumerate(dataloader):
            # start_time = time.time()
            data = common.to_cuda(data)
            image_path = data[constants.KEY_IMAGE_PATH]

            with torch.no_grad():
                prediction = model(data)
            # duration_time = time.time() - start_time

            if self.feat_vis:
                featmaps_dict = model.get_feat()
                from utils.visualizer import FeatVisualizer
                feat_visualizer = FeatVisualizer()
                feat_visualizer.visualize_maps(featmaps_dict)

            # initialize dets for each classes
            # dets = [[] for class_ind in range(self.n_classes)]

            scores = prediction[constants.KEY_CLASSES]
            boxes_2d = prediction[constants.KEY_BOXES_2D]
            dims = prediction[constants.KEY_DIMS]
            center_depth = prediction[constants.KEY_CENTER_DEPTH]
            dims = torch.cat([dims, center_depth], dim=-1)
            corners_2d = prediction[constants.KEY_CORNERS_2D]

            p2 = data[constants.KEY_STEREO_CALIB_P2_ORIG]

            # rcnn_3d = prediction['rcnn_3d']
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
                corners_2d_per_img = corners_2d[batch_ind]
                p2_per_img = p2[batch_ind]

                num_cols = corners_2d.shape[-1]
                dets = [np.zeros((0, 8, num_cols), dtype=np.float32)]
                dets_2d = [np.zeros((0, 5), dtype=np.float32)]
                dets_depth = []
                dets_dims = []

                for class_ind in range(1, self.n_classes):
                    # cls thresh
                    inds = torch.nonzero(
                        scores_per_img[:, class_ind] > self.thresh).view(-1)
                    threshed_scores_per_img = scores_per_img[inds, class_ind]
                    if inds.numel() > 0:
                        # if self.class_agnostic:
                        threshed_boxes_2d_per_img = boxes_2d_per_img[inds]
                        threshed_dims_per_img = dims_per_img[inds]
                        threshed_corners_2d_per_img = corners_2d_per_img[inds]
                        # threshed_rcnn_3d_per_img = rcnn_3d_per_img[inds]
                        # else:
                        # threshed_boxes_2d_per_img = boxes_2d_per_img[
                        # inds, class_ind * 4:class_ind * 4 + 4]
                        # concat boxes and scores
                        threshed_dets_per_img = torch.cat(
                            [
                                threshed_boxes_2d_per_img,
                                threshed_scores_per_img.unsqueeze(-1),
                                threshed_dims_per_img,
                            ],
                            dim=-1)

                        # sort by scores
                        _, order = torch.sort(threshed_scores_per_img, 0, True)
                        threshed_dets_per_img = threshed_dets_per_img[order]
                        threshed_corners_2d_per_img = threshed_corners_2d_per_img[
                            order]

                        # nms
                        keep = nms(threshed_dets_per_img[:, :4],
                                   threshed_dets_per_img[:, 4],
                                   self.nms).view(-1).long()
                        nms_dets_per_img = threshed_dets_per_img[
                            keep].detach().cpu().numpy()
                        nms_corners_2d_per_img = threshed_corners_2d_per_img[
                            keep].detach().cpu().numpy()

                        dets.append(nms_corners_2d_per_img)
                        dets_2d.append(nms_dets_per_img[:, :5])
                        dets_dims.append(nms_dets_per_img[:, 5:8])
                        dets_depth.append(nms_dets_per_img[:, 8:])
                    else:
                        dets.append(
                            np.zeros((0, 8, num_cols), dtype=np.float32))
                        dets_2d.append(np.zeros((0, 5)))
                        dets_dims.append(np.zeros((0, 3), dtype=np.float32))
                        dets_depth.append(np.zeros((0, 1), dtype=np.float32))

                corners = np.concatenate(dets, axis=0)
                dets_2d = np.concatenate(dets_2d, axis=0)
                dets_dims = np.concatenate(dets_dims, axis=0)
                dets_depth = np.concatenate(dets_depth, axis=0)
                corners_2d = None
                corners_3d = None
                if num_cols == 3:
                    corners_3d = corners
                else:
                    corners_2d = corners

                self.visualizer.render_image_corners_2d(
                    image_path[0],
                    boxes_2d=dets_2d,
                    corners_2d=corners_2d,
                    corners_3d=corners_3d,
                    p2=p2_per_img.cpu().numpy())

                duration_time = time.time() - end_time
                label_path = self._generate_label_path(image_path[batch_ind],
                                                       './results/corners')

                self.save_pseudo_3d(dets_2d, corners, dets_dims, dets_depth,
                                    label_path)
                sys.stdout.write('\r{}/{},duration: {}'.format(
                    step + 1, num_samples, duration_time))
                sys.stdout.flush()

                end_time = time.time()

    def test_3d(self, dataloader, model, logger):
        self.logger.info('Start testing')
        num_samples = len(dataloader)

        if self.feat_vis:
            # enable it before forward pass
            model.enable_feat_vis()
        end_time = 0

        for step, data in enumerate(dataloader):
            # start_time = time.time()
            data = common.to_cuda(data)
            image_path = data[constants.KEY_IMAGE_PATH]

            with torch.no_grad():
                prediction = model(data)
            # duration_time = time.time() - start_time

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

            # rcnn_3d = prediction['rcnn_3d']
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
                # rcnn_3d_per_img = rcnn_3d[batch_ind]
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
                        # threshed_rcnn_3d_per_img = rcnn_3d_per_img[inds]
                        # else:
                        # threshed_boxes_2d_per_img = boxes_2d_per_img[
                        # inds, class_ind * 4:class_ind * 4 + 4]
                        # concat boxes and scores
                        threshed_dets_per_img = torch.cat(
                            [
                                threshed_boxes_2d_per_img,
                                threshed_scores_per_img.unsqueeze(-1),
                                threshed_dims_per_img,
                                threshed_orients_per_img.unsqueeze(-1)
                            ],
                            dim=-1)

                        # sort by scores
                        _, order = torch.sort(threshed_scores_per_img, 0, True)
                        threshed_dets_per_img = threshed_dets_per_img[order]
                        # threshed_rcnn_3d_per_img = threshed_rcnn_3d_per_img[order]

                        # nms
                        keep = nms(threshed_dets_per_img[:, :4],
                                   threshed_dets_per_img[:, 4],
                                   self.nms).view(-1).long()
                        nms_dets_per_img = threshed_dets_per_img[
                            keep].detach().cpu().numpy()
                        # nms_rcnn_3d_per_img = threshed_rcnn_3d_per_img[keep].detach().cpu().numpy()

                        # calculate location
                        location = geometry_utils.calc_location(
                            nms_dets_per_img[:, 5:8], nms_dets_per_img[:, :5],
                            nms_dets_per_img[:, 8],
                            p2_per_img.cpu().numpy())
                        # import ipdb
                        # ipdb.set_trace()
                        # location, _ = mono_3d_postprocess_bbox(
                        # nms_rcnn_3d_per_img, nms_dets_per_img[:, :5],
                        # p2_per_img.cpu().numpy())
                        nms_dets_per_img = np.concatenate(
                            [
                                nms_dets_per_img[:, :5],
                                nms_dets_per_img[:, 5:8], location,
                                nms_dets_per_img[:, -1:]
                            ],
                            axis=-1)
                        # nms_dets_per_img = np.concatenate(
                        # [nms_dets_per_img[:, :5], location], axis=-1)

                        dets.append(nms_dets_per_img)
                    else:
                        dets.append([])

                duration_time = time.time() - end_time
                label_path = self._generate_label_path(image_path[batch_ind])
                self.save_mono_3d_dets(dets, label_path)
                sys.stdout.write('\r{}/{},duration: {}'.format(
                    step + 1, num_samples, duration_time))
                sys.stdout.flush()

                end_time = time.time()

    def test_2d(self, dataloader, model, logger):
        self.logger.info('Start testing')
        num_samples = len(dataloader)

        if self.feat_vis:
            # enable it before forward pass
            model.enable_feat_vis()
        end_time = 0

        for step, data in enumerate(dataloader):
            # start_time = time.time()
            data = common.to_cuda(data)
            image_path = data[constants.KEY_IMAGE_PATH]

            with torch.no_grad():
                prediction = model(data)
            # duration_time = time.time() - start_time

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
                for class_ind in range(1, self.n_classes):
                    # cls thresh
                    inds = torch.nonzero(
                        scores_per_img[:, class_ind] > self.thresh).view(-1)
                    threshed_scores_per_img = scores_per_img[inds, class_ind]
                    if inds.numel() > 0:
                        # if self.class_agnostic:
                        threshed_boxes_2d_per_img = boxes_2d_per_img[inds]
                        # else:
                        # threshed_boxes_2d_per_img = boxes_2d_per_img[
                        # inds, class_ind * 4:class_ind * 4 + 4]
                        # concat boxes and scores
                        threshed_dets_per_img = torch.cat(
                            [
                                threshed_boxes_2d_per_img,
                                threshed_scores_per_img.unsqueeze(-1),
                            ],
                            dim=-1)

                        # sort by scores
                        _, order = torch.sort(threshed_scores_per_img, 0, True)
                        threshed_dets_per_img = threshed_dets_per_img[order]

                        # nms
                        keep = nms(threshed_dets_per_img[:, :4],
                                   threshed_dets_per_img[:, 4],
                                   self.nms).view(-1).long()
                        nms_dets_per_img = threshed_dets_per_img[
                            keep].detach().cpu().numpy()

                        dets.append(nms_dets_per_img)
                    else:
                        dets.append([])

                duration_time = time.time() - end_time
                label_path = self._generate_label_path(image_path[batch_ind])
                self.save_dets(dets, label_path, image_path[batch_ind])
                sys.stdout.write('\r{}/{},duration: {}'.format(
                    step + 1, num_samples, duration_time))
                sys.stdout.flush()

                end_time = time.time()

    def test_super_nms(self, dataloader, model, logger):
        self.logger.info('Start testing')
        num_samples = len(dataloader)

        if self.feat_vis:
            # enable it before forward pass
            model.enable_feat_vis()
        end_time = 0

        for step, data in enumerate(dataloader):
            # start_time = time.time()
            data = common.to_cuda(data)
            image_path = data[constants.KEY_IMAGE_PATH]

            with torch.no_grad():
                prediction = model(data)
            # duration_time = time.time() - start_time

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

            batch_size = scores.shape[0]
            # scores = scores.view(-1, self.n_classes)
            # new_scores = torch.zeros_like(scores)
            # _, scores_argmax = scores.max(dim=-1)
            # row = torch.arange(0, scores_argmax.numel()).type_as(scores_argmax)
            # new_scores[row, scores_argmax] = scores[row, scores_argmax]
            # scores = new_scores.view(batch_size, -1, self.n_classes)

            #  if step == 6:
            #  import ipdb
            #  ipdb.set_trace()

            for batch_ind in range(batch_size):
                boxes_2d_per_img = boxes_2d[batch_ind]
                scores_per_img = scores[batch_ind]
                for class_ind in range(1, self.n_classes):
                    # cls thresh
                    # import ipdb
                    # ipdb.set_trace()
                    inds = torch.nonzero(
                        scores_per_img[:, class_ind] > 0.01).view(-1)
                    threshed_scores_per_img = scores_per_img[inds, class_ind]
                    if inds.numel() > 0:
                        # if self.class_agnostic:
                        threshed_boxes_2d_per_img = boxes_2d_per_img[inds]
                        # else:
                        # threshed_boxes_2d_per_img = boxes_2d_per_img[
                        # inds, class_ind * 4:class_ind * 4 + 4]
                        # concat boxes and scores
                        threshed_dets_per_img = torch.cat(
                            [
                                threshed_boxes_2d_per_img,
                                threshed_scores_per_img.unsqueeze(-1),
                            ],
                            dim=-1)

                        # sort by scores
                        _, order = torch.sort(threshed_scores_per_img, 0, True)
                        threshed_dets_per_img = threshed_dets_per_img[order]

                        # nms
                        # keep = nms(threshed_dets_per_img[:, :4],
                        # threshed_dets_per_img[:, 4],
                        # self.nms).view(-1).long()
                        keep = box_ops.super_nms(
                            threshed_dets_per_img[:, :4],
                            0.8,
                            nms_num=3,
                            loop_time=2)
                        nms_dets_per_img = threshed_dets_per_img[
                            keep].detach().cpu().numpy()

                        dets.append(nms_dets_per_img)
                    else:
                        dets.append([])

                duration_time = time.time() - end_time
                label_path = self._generate_label_path(image_path[batch_ind])
                self.save_dets(dets, label_path)
                sys.stdout.write('\r{}/{},duration: {}'.format(
                    step + 1, num_samples, duration_time))
                sys.stdout.flush()

                end_time = time.time()

    def test_pc(self, dataloader, model, logger):
        self.logger.info('Start testing')
        num_samples = len(dataloader)

        if self.feat_vis:
            # enable it before forward pass
            model.enable_feat_vis()
        end_time = 0

        for step, data in enumerate(dataloader):
            # start_time = time.time()
            data = common.to_cuda(data)
            image_path = data[constants.KEY_IMAGE_PATH]

            with torch.no_grad():
                prediction = model(data)
            # duration_time = time.time() - start_time

            if self.feat_vis:
                featmaps_dict = model.get_feat()
                from utils.visualizer import FeatVisualizer
                feat_visualizer = FeatVisualizer()
                feat_visualizer.visualize_maps(featmaps_dict)

            # initialize dets for each classes
            # dets = [[] for class_ind in range(self.n_classes)]

            scores = prediction[constants.KEY_CLASSES]
            boxes_2d = prediction[constants.KEY_BOXES_2D]
            depth_map = prediction[constants.KEY_DEPTHMAP]
            # import ipdb
            # ipdb.set_trace()
            #  import ipdb
            #  ipdb.set_trace()
            p2 = data[constants.KEY_STEREO_CALIB_P2_ORIG]

            # rcnn_3d = prediction['rcnn_3d']
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
                #  dims_per_img = dims[batch_ind]
                depth_map_per_img = depth_map[batch_ind]
                p2_per_img = p2[batch_ind]

                num_cols = depth_map.shape[-1]
                dets = [np.zeros((0, num_cols), dtype=np.float32)]
                dets_2d = [np.zeros((0, 4), dtype=np.float32)]

                for class_ind in range(1, self.n_classes):
                    # cls thresh
                    inds = torch.nonzero(
                        scores_per_img[:, class_ind] > self.thresh).view(-1)
                    threshed_scores_per_img = scores_per_img[inds, class_ind]
                    if inds.numel() > 0:
                        # if self.class_agnostic:
                        threshed_boxes_2d_per_img = boxes_2d_per_img[inds]
                        #  threshed_dims_per_img = dims_per_img[inds]
                        threshed_depth_map_per_img = depth_map_per_img[inds]
                        # threshed_rcnn_3d_per_img = rcnn_3d_per_img[inds]
                        # else:
                        # threshed_boxes_2d_per_img = boxes_2d_per_img[
                        # inds, class_ind * 4:class_ind * 4 + 4]
                        # concat boxes and scores
                        threshed_dets_per_img = torch.cat(
                            [
                                threshed_boxes_2d_per_img,
                                threshed_scores_per_img.unsqueeze(-1),
                                #  threshed_dims_per_img,
                            ],
                            dim=-1)

                        # sort by scores
                        _, order = torch.sort(threshed_scores_per_img, 0, True)
                        threshed_dets_per_img = threshed_dets_per_img[order]
                        threshed_depth_map_per_img = threshed_depth_map_per_img[
                            order]

                        # nms
                        keep = nms(threshed_dets_per_img[:, :4],
                                   threshed_dets_per_img[:, 4],
                                   self.nms).view(-1).long()
                        nms_dets_per_img = threshed_dets_per_img[
                            keep].detach().cpu().numpy()
                        nms_depth_map_per_img = threshed_depth_map_per_img[
                            keep].detach().cpu().numpy()

                        dets.append(nms_depth_map_per_img)
                        dets_2d.append(nms_dets_per_img[:, :4])
                    else:
                        dets.append(np.zeros((0, num_cols), dtype=np.float32))
                        dets_2d.append(np.zeros((0, 4)))

                # import ipdb
                # ipdb.set_trace()
                instance_depth_map = np.concatenate(dets, axis=0)
                dets_2d = np.concatenate(dets_2d, axis=0)
                pointcloud_utils.visualize_pointcloud(image_path[0], dets_2d,
                                                      instance_depth_map,
                                                      p2_per_img.cpu().numpy())

                # self.visualizer.render_image_and_pointcloud(
                # image_path[0],
                # boxes_2d=dets_2d,
                # depth_maps=depth_maps,
                # p2=p2_per_img.cpu().numpy())

                duration_time = time.time() - end_time
                #  label_path = self._generate_label_path(image_path[batch_ind])
                #  self.save_mono_3d_dets(dets, label_path)
                sys.stdout.write('\r{}/{},duration: {}'.format(
                    step + 1, num_samples, duration_time))
                sys.stdout.flush()

                end_time = time.time()

    def test(self, dataloader, model, logger):
        test_fn = getattr(self, self.test_type)
        test_fn(dataloader, model, logger)
        # self.test_super_nms(dataloader, model, logger)
        #  self.test_2d(dataloader, model, logger)
        #  self.test_3d(dataloader, model, logger)
        # self.test_corners_3d(dataloader, model, logger)
