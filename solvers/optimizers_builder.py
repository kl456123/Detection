# -*- coding: utf-8 -*-
from utils.registry import OPTIMIZERS

from torch import optim


def make_optim_params(model, config, logger=None):
    """
    get learnable params ,lr and weight decay
    """
    # set up logger
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    all_params = model.named_parameters()
    # learnable_params = []
    optim_params = []
    for name, value in all_params:
        if value.requires_grad:
            # learnable_params.append(param)
            logger.info('{} is learnable!'.format(name))
            if 'bias' in name:
                lr = config['base_lr'] * config['bias_lr_factor']
                weight_decay = config['weight_decay_bias']
                param = {
                    'params': [value],
                    'lr': lr,
                    'weight_decay': weight_decay
                }
                optim_params.append(param)
        else:
            logger.info('{} is not learnable'.format(name))

    return optim_params


@OPTIMIZERS.register('sgd')
def build_sgd(optimizer_config, model, logger=None):
    optim_params = make_optim_params(model, optimizer_config, logger)
    sgd = optim.SGD(optim_params,
                    optimizer_config['base_lr'],
                    momentum=optimizer_config['monmentum'])
    return sgd


@OPTIMIZERS.register('adam')
def build_adam(optimizer_config, model, logger=None):
    optim_params = make_optim_params(model, optimizer_config, logger)
    adam = optim.Adam(
        optim_params,
        optimizer_config['base_lr'],
        beta=optimizer_config['beta'],
        eps=optimizer_config['eps'])
    return adam
