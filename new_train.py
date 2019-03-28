import os

from core.utils.summary_writer import SummaryWriter
from core.utils.saver import Saver
from core.utils.logger import setup_logger

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
    parser.add_argument('--logger_level', default='INFO',
                        type=str, help='logger level')
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

    scheduler = builder.build_scheduler(
        train_config['scheduler_config'])

    # some components for logging and saving(saver and summaryer)
    output_dir = os.path.join(train_config['save_dir'], model_config[
        'net'],  data_config['name'])
    saver = Saver(output_dir)

    summary_path = os.path.join(output_dir, './summary')
    summary_writer = SummaryWriter(summary_path)

    trainer.train(dataloader, model, optimizer,
                  scheduler, saver, summary_writer)


def generate_config(args, logger):
    pass


if __name__ == '__main__':
    # first setup logger
    logger = setup_logger('detection', args.logger_level)

    config = generate_config(args, logger)
    train(config, logger)
