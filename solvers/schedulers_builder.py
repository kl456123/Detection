# -*- coding: utf-8 -*-

from utils.registry import SCHEDULERS
from torch.optim import lr_scheduler
from .schedulers.multi_step import WarmupMultiStepLR


@SCHEDULERS.register('step')
def build_step_scheduler(config, optimizer):
    lr_decay_step = config['lr_decay_step']
    lr_decay_gamma = config['lr_decay_gamma']
    return lr_scheduler.StepLR(
        optimizer,
        lr_decay_step,
        lr_decay_gamma,
        last_epoch=config['last_step'])


@SCHEDULERS.register('multi_step')
def build_multistep_scheduler(config, optimizer):
    return WarmupMultiStepLR(
        optimizer,
        config['milestones'],
        gamma=config['gamma'],
        warmup_factor=config['warmup_factor'],
        warmup_iters=config['warmup_iters'],
        warmup_method=config['warmup_method'],
        last_epoch=config['last_epoch'])
