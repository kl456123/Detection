# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick # --------------------------------------------------------

import sys
sys.path.append('./lib')
import os
import numpy as np
import argparse
import pprint
import time
import json

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from core.saver import Saver
from core import tester
from builder.dataloader_builders.kitti_dataloader_builder import KittiDataLoaderBuilder
from builder import model_builder


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


def infer_config_fn(args):
    import glob
    output_dir = args.load_dir + '/' + args.net + '/' + args.dataset
    possible_config = glob.glob(os.path.join(output_dir, '*.json'))
    assert len(possible_config) == 1
    return os.path.join(output_dir, possible_config[0])


if __name__ == '__main__':
    args = parse_args()
    # assert args.config is not None, 'please select a config file(json)'
    if args.config is None:
        # infer it
        args.config = infer_config_fn(args)
    with open(args.config) as f:
        config = json.load(f)

    model_config = config['model_config']
    data_config = config['eval_data_config']
    eval_config = config['eval_config']

    model_config['pretrained'] = False
    model_config['target_assigner_config'][
        'fake_match_thresh'] = args.fake_match_thresh

    assert args.net is not None, 'please select a base model'
    model_config['net'] = args.net

    eval_config['feat_vis'] = args.feat_vis

    if args.dataset is not None:
        data_config['name'] = args.dataset

    eval_config['rois_vis'] = args.rois_vis
    eval_config['use_which_result'] = args.use_which_result
    data_config['dataset_config']['use_gt'] = args.use_gt

    if args.nms is not None:
        eval_config['nms'] = args.nms

    if args.thresh is not None:
        eval_config['thresh'] = args.thresh

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

    eval_out = eval_config['eval_out']
    if not os.path.exists(eval_out):
        os.makedirs(eval_out)
    else:
        print('dir {} exist already!'.format(eval_out))

    #restore from random or checkpoint
    restore = True
    # two methods to load model
    # 1. load from training dir
    # 2. load from any other dirs,it just needs config and model path
    if args.checkepoch is not None and args.checkpoint is not None:
        checkpoint_name = 'faster_rcnn_{}_{}.pth'.format(args.checkepoch,
                                                         args.checkpoint)

        assert args.load_dir is not None, 'please choose a directory to load checkpoint'
        eval_config['load_dir'] = args.load_dir
        input_dir = eval_config['load_dir'] + "/" + model_config[
            'net'] + "/" + data_config['name']
        if not os.path.exists(input_dir):
            raise Exception(
                'There is no input directory for loading network from {}'.
                format(input_dir))
    elif args.model is not None:
        # assert args.model is not None, 'please determine model or checkpoint'
        # it should be a path to model
        checkpoint_name = os.path.basename(args.model)
        input_dir = os.path.dirname(args.model)
    else:
        restore = False

    # log for restore
    if restore:
        print("restore from checkpoint")
    else:
        print("use pytorch default initialization")

    # model
    fasterRCNN = model_builder.build(model_config, training=False)

    if restore:
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
