import os

from core.utils.summary_writer import SummaryWriter
from core.saver import Saver
from core.utils.logger import setup_logger
from core.utils.config import Config

from core.trainer import Trainer
from data import dataloaders
from models import detectors
from solvers import optimizers
from solvers import schedulers

import numpy as np
import argparse
import shutil
import torch
from core.utils import common


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


def train(config, logger):
    data_config = config['data_config']
    model_config = config['model_config']
    train_config = config['train_config']

    # build model
    model = detectors.build(model_config)

    # move to gpus before building optimizer
    if train_config['mGPUs']:
        model = common.MyParallel(model, device_ids=[0, 1])

    if train_config['cuda']:
        model = model.cuda()

    # build optimizer and scheduler
    optimizer = optimizers.build(train_config['optimizer_config'], model)

    # force to change lr before scheduler
    if train_config['lr']:
        common.change_lr(optimizer, train_config['lr'])

    scheduler = schedulers.build(train_config['scheduler_config'], optimizer)

    # some components for logging and saving(saver and summaryer)
    output_dir = os.path.join(train_config['output_path'],
                              model_config['type'], data_config['name'])
    saver = Saver(output_dir)

    # resume
    if train_config['resume']:
        checkpoint_path = '{}.pth'.format(train_config['checkpoint'])
        logger.info(
            'resume from checkpoint detector_{}'.format(checkpoint_path))
        params_dict = {
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'num_iters': None
        }

        saver.load(params_dict, checkpoint_path)
        train_config['num_iters'] = params_dict['num_iters']

    # build dataloader after resume(may be or not)
    # dataloader = dataloaders.build(data_config)
    dataloader = dataloaders.make_data_loader(data_config)

    # use model to initialize
    if train_config['model']:
        model_path = train_config['model']
        logger.info('initialize model from {}'.format(model_path))
        params_dict = {'model': model}
        saver.load(params_dict, model_path)

    summary_path = os.path.join(output_dir, './summary')
    logger.info('setup summary_dir: {}'.format(summary_path))
    summary_writer = SummaryWriter(summary_path)

    logger.info('setup trainer')
    trainer = Trainer(train_config, logger)
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
    model_config['type'] = args.net

    # output dir
    output_dir = os.path.join(train_config['output_path'],
                              model_config['type'], data_config['name'])
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

    train_config['checkpoint'] = args.checkpoint

    return config


if __name__ == '__main__':
    args = parse_args()
    # first setup logger
    logger = setup_logger('detection')

    config = generate_config(args, logger)
    train(config, logger)
