# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import numpy as np
import argparse
import pprint
import time

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from configs import kitti_config
from core.saver import Saver
from core import tester
from builder.dataloader_builders.kitti_dataloader_builder import KittiDataLoaderBuilder


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        default='cfgs/vgg16.yml',
        type=str)
    parser.add_argument(
        '--set',
        dest='set_cfgs',
        help='set config keys',
        default=None,
        nargs=argparse.REMAINDER)
    parser.add_argument(
        '--load_dir',
        dest='load_dir',
        help='directory to load models',
        default="/srv/share/jyang375/models",
        type=str)
    parser.add_argument(
        '--cuda', dest='cuda', help='whether use CUDA', action='store_true')
    parser.add_argument(
        '--mGPUs',
        dest='mGPUs',
        help='whether use multiple GPUs',
        action='store_true')
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
        default=1,
        type=int)
    parser.add_argument(
        '--checkpoint',
        dest='checkpoint',
        help='checkpoint to load network',
        default=10021,
        type=int)
    parser.add_argument(
        '--vis', dest='vis', help='visualization mode', action='store_true')

    parser.add_argument(
        '--img_path',
        dest='img_path',
        help='path to image',
        default='',
        type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    model_config = kitti_config.model_config
    data_config = kitti_config.eval_data_config
    eval_config = kitti_config.eval_config

    model_config['pretrained'] = False

    if args.img_path:
        dataset_config = data_config['dataset_config']
        # disable dataset file,just use image directly
        dataset_config['dataset_file'] = None
        dataset_config['demo_file'] = args.img_path

    print('Called with args:')
    print(args)

    np.random.seed(eval_config['rng_seed'])

    print('Using config:')
    pprint.pprint({
        'model_config': model_config,
        'data_config': data_config,
        'eval_config': eval_config
    })

    input_dir = eval_config['load_dir'] + "/" + model_config[
        'net'] + "/" + data_config['name']
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from '
                        + input_dir)

    eval_out = eval_config['eval_out']
    if not os.path.exists(eval_out):
        os.makedirs(eval_out)
    else:
        print('dir {} exist already!'.format(eval_out))

    checkpoint_name = 'faster_rcnn_{}_{}.pth'.format(args.checkepoch,
                                                     args.checkpoint)

    # model
    fasterRCNN = resnet(model_config)
    fasterRCNN.eval()
    fasterRCNN.create_architecture()

    # saver
    saver = Saver(input_dir)
    saver.load({'model': fasterRCNN}, checkpoint_name)

    if args.cuda:
        fasterRCNN.cuda()

    start = time.time()

    vis = args.vis

    data_loader_builder = KittiDataLoaderBuilder(data_config, training=False)
    data_loader = data_loader_builder.build()

    tester.test(eval_config, data_loader, fasterRCNN)
