# -*- coding: utf-8 -*-
"""
python infer.py --load_dir /data/object/liangxiong/fpn_mono_3d --checkpoint 516000 --net fpn_mono_3d --dataset kitti
python infer.py --model /data/object/liangxiong/fpn_mono_3d/fpn_mono_3d/kitti/detector_516000.pth \
                --config /data/object/liangxiong/fpn_mono_3d/fpn_mono_3d/kitti/fpn_mono_3d_kitti_config.json \
                --net fpn_mono_3d
"""

import os
import random
import numpy as np
from core.utils.config import Config
import torch
from core.utils.logger import setup_logger
from lib.model.roi_layers import nms
import pprint
import argparse

from core.saver import Saver
from core.tester import Tester
from utils import geometry_utils

from data import dataloaders
from models import detectors
from core import constants
from core.utils import common
from data import transforms

from utils.drawer import ImageVisualizer

import cv2
from PIL import Image


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument(
        '--load_dir',
        dest='load_dir',
        help='directory to load models',
        default="/srv/share/jyang375/models",
        type=str)
    parser.add_argument(
        '--calib_file',
        dest='calib_file',
        help='calibration file using kitti format',
        default='',
        type=str)
    parser.add_argument(
        '--calib_dir',
        dest='calib_dir',
        help='calibration directory using kitti format',
        default='',
        type=str)
    parser.add_argument(
        '--cuda', dest='cuda', help='whether use CUDA', action='store_true')
    parser.add_argument(
        '--mGPUs',
        dest='mGPUs',
        help='whether use multiple GPUs',
        action='store_true')
    parser.add_argument(
        '--net', dest='net', help='which base mode to use', type=str)
    parser.add_argument(
        '--parallel_type',
        dest='parallel_type',
        help='which part of model to parallel, 0: all, 1: model before roi pooling',
        default=0,
        type=int)
    parser.add_argument(
        '--checkepoch',
        dest='checkepoch',
        help='checkepoch to load network',
        type=int)
    parser.add_argument(
        '--checkpoint',
        dest='checkpoint',
        help='checkpoint to load network',
        type=int)
    parser.add_argument(
        '--vis', dest='vis', help='visualization mode', action='store_true')

    parser.add_argument(
        '--img_path',
        dest='img_path',
        help='path to image',
        default='',
        type=str)
    parser.add_argument(
        '--img_dir',
        dest='img_dir',
        help='img directory',
        default='',
        type=str)
    parser.add_argument(
        '--rois_vis',
        dest='rois_vis',
        help='if to visualize rois',
        action='store_true')
    parser.add_argument(
        '--feat_vis',
        dest='feat_vis',
        help='visualize feat or not',
        default=False,
        type=bool)
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='kitti or others',
        type=str,
        default='kitti')
    parser.add_argument(
        '--config', dest='config', help='config file(.json)', type=str)
    parser.add_argument(
        "--nms", dest='nms', help='nms to suppress bbox', type=float)
    parser.add_argument(
        "--thresh", dest="thresh", help='thresh for scores', type=float)
    parser.add_argument(
        "--model", dest="model", help="path to checkpoint", type=str)
    parser.add_argument(
        "--use_which_result",
        dest="use_which_result",
        help="use rpn results to leading the output",
        type=str,
        default='none')
    parser.add_argument(
        "--fake_match_thresh",
        dest="fake_match_thresh",
        help="eval the performance of bbox",
        type=float,
        default=0.7)
    parser.add_argument(
        "--use_gt",
        dest="use_gt",
        help='whether to use gt for analysis',
        type=bool,
        default=False)

    args = parser.parse_args()
    return args


class Mono3DInfer(object):
    KITTI_MEAN_DIMS = {
        'Car': [3.88311640418, 1.62856739989, 1.52563191462],
        'Van': [5.06763659, 1.9007158, 2.20532825],
        'Truck': [10.13586957, 2.58549199, 3.2520595],
        'Pedestrian': [0.84422524, 0.66068622, 1.76255119],
        'Person_sitting': [0.80057803, 0.5983815, 1.27450867],
        'Cyclist': [1.76282397, 0.59706367, 1.73698127],
        'Tram': [16.17150617, 2.53246914, 3.53079012],
        'Misc': [3.64300781, 1.54298177, 1.92320313]
    }

    def get_random_color(self):
        color_code = []
        for _ in range(3):
            color_code.append(random.randint(0, 255))
        return color_code

    def __init__(self, args):
        # first setup logger
        self.logger = setup_logger()
        self.args = args

        self.config = self.generate_config(args, self.logger)

        self.data_config = self.config['eval_data_config']

        self.dataset_config = self.data_config['dataset_config']

        self.classes = ['bg'] + self.dataset_config['classes']
        self.n_classes = len(self.classes)

        colors = []
        for i in range(self.n_classes):
            colors.append(self.get_random_color())

        self.eval_config = self.config['eval_config']
        self.thresh = self.eval_config['thresh']
        self.nms = self.eval_config['nms']

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

        self.visualizer.colors = colors
        self.visualizer.classes = self.classes

    def preprocess(self, im, stereo_calib_p2):
        """
            Convert image to data dict
        """

        image_input = im
        image_shape = image_input.size[::-1]
        # no scale now
        image_scale = (1.0, 1.0)
        image_info = image_shape + image_scale

        # as for calib, it can read from different files
        # for each sample or single file for all samples

        transform_sample = {}
        transform_sample[constants.KEY_IMAGE] = image_input
        transform_sample[
            constants.KEY_STEREO_CALIB_P2] = stereo_calib_p2.astype(np.float32)

        # (h,w,scale)
        transform_sample[constants.KEY_IMAGE_INFO] = np.asarray(
            image_info, dtype=np.float32)

        mean_dims = self._generate_mean_dims()
        transform_sample[constants.KEY_MEAN_DIMS] = mean_dims

        transform_sample[constants.KEY_STEREO_CALIB_P2_ORIG] = np.copy(
            transform_sample[constants.KEY_STEREO_CALIB_P2])

        # transform
        transform_config = self.data_config['transform_config']
        transform = transforms.build(transform_config)

        training_sample = transform(transform_sample)

        return training_sample

    def _generate_mean_dims(self):
        mean_dims = []
        for class_type in self.classes[1:]:
            mean_dims.append(self.KITTI_MEAN_DIMS[class_type][::-1])
        return np.stack(mean_dims, axis=0).astype(np.float32)

    def to_batch(self, data):
        #  import ipdb
        #  ipdb.set_trace()
        for key in data:
            data[key] = data[key][None, ...]
        return data

    def inference(self, im, p2):
        """
        Args:
            im: shape(N, 3, H, W)

        Returns:
            dets: shape(N, M, 8)
        """
        config = self.config
        args = self.args
        eval_config = config['eval_config']
        model_config = config['model_config']
        data_config = config['eval_data_config']

        np.random.seed(eval_config['rng_seed'])

        self.logger.info('Using config:')
        pprint.pprint({
            'model_config': model_config,
            'data_config': data_config,
            'eval_config': eval_config
        })

        eval_out = eval_config['eval_out']
        if not os.path.exists(eval_out):
            self.logger.info('creat eval out directory {}'.format(eval_out))
            os.makedirs(eval_out)
        else:
            self.logger.warning('dir {} exist already!'.format(eval_out))

        # restore from random or checkpoint
        restore = True
        # two methods to load model
        # 1. load from any other dirs,it just needs config and model path
        # 2. load from training dir
        if args.model is not None:
            # assert args.model is not None, 'please determine model or checkpoint'
            # it should be a path to model
            checkpoint_name = os.path.basename(args.model)
            input_dir = os.path.dirname(args.model)
        elif args.checkpoint is not None:
            checkpoint_name = 'detector_{}.pth'.format(args.checkpoint)
            assert args.load_dir is not None, 'please choose a directory to load checkpoint'
            eval_config['load_dir'] = args.load_dir
            input_dir = os.path.join(eval_config['load_dir'],
                                     model_config['type'], data_config['name'])
            if not os.path.exists(input_dir):
                raise Exception(
                    'There is no input directory for loading network from {}'.
                    format(input_dir))
        else:
            restore = False

        # log for restore
        if restore:
            self.logger.info("restore from checkpoint")
        else:
            self.logger.info("use pytorch default initialization")

        # model
        model = detectors.build(model_config)
        model.eval()

        if restore:
            # saver
            saver = Saver(input_dir)
            saver.load({'model': model}, checkpoint_name)

        model = model.cuda()

        #  dataloader = dataloaders.make_data_loader(data_config, training=False)

        self.logger.info('Start testing')
        #  num_samples = len(dataloader)

        #  for step, data in enumerate(dataloader):
        data = self.preprocess(im, p2)
        data = self.to_batch(data)
        data = common.to_cuda(data)
        #  image_path = data[constants.KEY_IMAGE_PATH]

        with torch.no_grad():
            prediction = model(data)

        # initialize dets for each classes
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

        boxes_2d_per_img = boxes_2d[0]
        scores_per_img = scores[0]
        dims_per_img = dims[0]
        orients_per_img = orients[0]
        p2_per_img = p2[0]
        # rcnn_3d_per_img = rcnn_3d[batch_ind]
        # import ipdb
        # ipdb.set_trace()
        for class_ind in range(1, self.n_classes):
            # cls thresh
            inds = torch.nonzero(
                scores_per_img[:, class_ind] > self.thresh).view(-1)
            threshed_scores_per_img = scores_per_img[inds, class_ind]
            if inds.numel() > 0:
                threshed_boxes_2d_per_img = boxes_2d_per_img[inds]
                threshed_dims_per_img = dims_per_img[inds]
                threshed_orients_per_img = orients_per_img[inds]
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
                nms_dets_per_img = threshed_dets_per_img[keep].detach().cpu(
                ).numpy()

                # calculate location
                location = geometry_utils.calc_location(
                    nms_dets_per_img[:, 5:8], nms_dets_per_img[:, :5],
                    nms_dets_per_img[:, 8], p2_per_img.cpu().numpy())

                nms_dets_per_img = np.concatenate(
                    [
                        nms_dets_per_img[:, :5], nms_dets_per_img[:, 5:8],
                        location, nms_dets_per_img[:, -1:]
                    ],
                    axis=-1)

                dets.append(nms_dets_per_img)
            else:
                dets.append([])

            #  duration_time = time.time() - end_time
            #  label_path = self._generate_label_path(image_path[batch_ind])
            #  self.save_mono_3d_dets(dets, label_path)
            #  sys.stdout.write('\r{}/{},duration: {}'.format(
            #  step + 1, num_samples, duration_time))
            #  sys.stdout.flush()

            #  end_time = time.time()

            #  xmin, ymin, xmax, ymax, cf, h, w, l, x, y, z, ry
        return dets

    def parse_kitti_format(self, dets, p2):
        """
        Args:
            dets: (N, 12)
        Returns:
            results: (boxes_3d, boxes_2d, label_classes, p2)
        """
        label_classes = []
        boxes_2d = []
        boxes_3d = []
        for cls_ind, det_per_cls in enumerate(dets):
            if len(det_per_cls) == 0:
                continue
            boxes_2d.append(det_per_cls[:, :5])
            boxes_3d.append(det_per_cls[:, [8, 9, 10, 5, 6, 7, 11, 4]])
            label_classes.extend([cls_ind] * det_per_cls.shape[0])

        p2 = p2
        boxes_2d = np.concatenate(boxes_2d, axis=0)
        boxes_3d = np.concatenate(boxes_3d, axis=0)
        label_classes = np.asarray(label_classes)

        return boxes_3d, boxes_2d, label_classes, p2

    def vis_result(self, im_to_show, dets, p2):
        """
        Args:
            im_to_show: shape(H, W, 3)
        """

        results = self.parse_kitti_format(dets, p2)
        image = self.visualizer.render_image(im_to_show, results)
        # self.visualizer.render_image_3d(im_to_show, dets, self.label_classes,
        # p2)
        # if self.online:
        # image postprocess
        cv2.imshow("test", image)
        cv2.waitKey(0)
        # else:
        # sample_name = self.get_sample_name_from_path(image_path)
        # saved_path = self.get_saved_path(sample_name)
        # cv2.imwrite(saved_path, image)

    def generate_config(self, args, logger):

        # read config from file
        if args.config is None:
            output_dir = os.path.join(args.load_dir, args.net, args.dataset)
            config_path = Config.infer_fromdir(output_dir)
        else:
            config_path = args.config
        config = Config.fromjson(config_path)

        eval_config = config['eval_config']
        model_config = config['model_config']
        data_config = config['eval_data_config']

        np.random.seed(eval_config['rng_seed'])

        torch.backends.cudnn.benchmark = True

        model_config['pretrained'] = False
        eval_config['feat_vis'] = args.feat_vis

        assert args.net is not None, 'please select a base model'
        model_config['type'] = args.net

        # use multi gpus to parallel
        eval_config['mGPUs'] = args.mGPUs
        eval_config['cuda'] = args.cuda

        # use pretrained model to initialize
        eval_config['model'] = args.model

        eval_config['checkpoint'] = args.checkpoint

        if args.nms is not None:
            eval_config['nms'] = args.nms

        if args.thresh is not None:
            eval_config['thresh'] = args.thresh
            model_config['score_thresh'] = args.thresh

        if args.img_path:
            dataset_config = data_config['dataset_config']
            # disable dataset file,just use image directly
            dataset_config['dataset_file'] = None
            dataset_config['demo_file'] = args.img_path
            dataset_config['calib_file'] = args.calib_file

        if args.img_dir:
            dataset_config = data_config['dataset_config']
            # disable dataset file,just use image directly
            dataset_config['dataset_file'] = None
            dataset_config['img_dir'] = args.img_dir

        if args.calib_file:
            dataset_config = data_config['dataset_config']
            dataset_config['calib_file'] = args.calib_file

        if args.calib_dir:
            dataset_config = data_config['dataset_config']
            dataset_config['calib_dir'] = args.calib_dir

        return config


def load_projection_matrix(calib_file):
    """Load the camera project matrix."""
    assert os.path.isfile(calib_file)
    with open(calib_file) as f:
        lines = f.readlines()
        line = lines[2]
        line = line.split()
        assert line[0] == 'P2:'
        p = [float(x) for x in line[1:]]
        p = np.array(p).reshape(3, 4)

    f.close()
    return p


def main():
    args = parse_args()
    infer = Mono3DInfer(args)
    image_path = '/data/dm202_3w/left_img/003000.png'
    calib_path = './000004.txt'
    im = Image.open(image_path)
    im_cv2 = cv2.imread(image_path)
    p2 = load_projection_matrix(calib_path)
    dets = infer.inference(im, p2)
    infer.vis_result(im_cv2, dets, p2)


if __name__ == '__main__':
    main()
