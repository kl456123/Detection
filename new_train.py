import os
import logging

from core.utils.summary_writer import SummaryWriter
from core.saver import Saver
from core.utils.logger import setup_logger
from core.utils.config import Config

from core import trainer
from utils import builder

import numpy as np
import argparse
import shutil
import json

import torch
import torch.nn as nn


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument(
        '--epochs',
        dest='max_epochs',
        help='number of epochs to train',
        default=20,
        type=int)
    parser.add_argument(
        '--cuda', dest='cuda', help='whether use CUDA', action='store_true')
    parser.add_argument(
        '--mGPUs',
        dest='mGPUs',
        help='whether use multiple GPUs',
        action='store_true')

    # resume trained model
    parser.add_argument(
        '--r',
        dest='resume',
        help='resume checkpoint or not',
        default=False,
        type=bool)
    parser.add_argument(
        '--net', dest='net', help='which base mode to use', type=str)
    parser.add_argument(
        '--checkepoch',
        dest='checkepoch',
        help='checkepoch to load model',
        default=1,
        type=int)
    parser.add_argument(
        '--checkpoint',
        dest='checkpoint',
        help='checkpoint to load model',
        default=0,
        type=int)
    # log and diaplay
    parser.add_argument(
        '--use_tfboard',
        dest='use_tfboard',
        help='whether use tensorflow tensorboard',
        default=False,
        type=bool)
    parser.add_argument(
        '--config', dest='config', help='config file(.json)', type=str)
    parser.add_argument('--lr', dest='lr', help='learning rate', type=float)
    parser.add_argument(
        '--model', dest='model', help='path to pretrained model', type=str)

    parser.add_argument(
        '--in_path', default=None, type=str, help='Input directory.')
    parser.add_argument(
        '--out_path', default=None, type=str, help='Output directory.')
    parser.add_argument(
        '--logger_level', default='INFO', type=str, help='logger level')
    args = parser.parse_args()
    return args


def change_lr(lr, optimizer):
    for param_group in optimizer.param_groups:
        # used for scheduler
        param_group['initial_lr'] = lr


def train(config, logger):
    data_config = config['data_config']
    model_config = config['model_config']
    train_config = config['train_config']

    # build dataloader
    dataloader = builder.build_dataloader(data_config)

    # build model
    model = builder.build_model()

    # build optimizer and scheduler
    optimizer = builder.build_optimizer(train_config['optimizer_config'])

    scheduler = builder.build_scheduler(train_config['scheduler_config'])

    # some components for logging and saving(saver and summaryer)
    output_dir = os.path.join(train_config['save_dir'], model_config['net'],
                              data_config['name'])
    saver = Saver(output_dir)

    summary_path = os.path.join(output_dir, './summary')
    summary_writer = SummaryWriter(summary_path)

    trainer.train(dataloader, model, optimizer, scheduler, saver,
                  summary_writer)


def generate_config(args, logger):
    # read config from file
    config = Config.fromjson(args.config)

    train_config = config['train_config']
    model_config = config['model_config']
    data_config = config['data_config']

    np.random.seed(train_config['rng_seed'])

    train_config['output_path'] = args.out_path

    torch.backends.cudnn.benchmark = True

    assert args.net is not None, 'please select a base model'
    model_config['net'] = args.net

    # output dir
    output_dir = os.path.join(train_config['output_path'], model_config['net'],
                              data_config['name'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info('create new directory {}'.format(output_dir))
    else:
        logger.info('output_dir is already exist')

    # copy config to output dir
    shutil.copy2(args.config, output_dir)

    # data input path
    if args.in_path is not None:
        # overwrite the data root path
        data_config['dataset_config']['root_path'] = args.in_path

    logger.info('checkpoint will be saved to {}'.format(output_dir))

    # use multi gpus to parallel
    train_config['mGPUs'] = args.mGPUs
    train_config['cuda'] = args.cuda

    # resume from checkpoint
    train_config['resume'] = args.resume

    # use pretrained model to initialize
    train_config['model'] = args.model

    # reset lr(modify the initial_lr of lr_scheduler)
    train_config['lr'] = args.lr

    return config


if __name__ == '__main__':
    args = parse_args()
    # first setup logger
    logger = setup_logger('detection')

    config = generate_config(args, logger)
    train(config, logger)
