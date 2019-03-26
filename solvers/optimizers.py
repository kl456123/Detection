# -*- coding: utf-8 -*-
from utils.registry import OPTIMIZERS


@OPTIMIZERS.register('sgd')
def build_sgd(optimizer_config, model):
    pass


@OPTIMIZERS.register('adam')
def build_adam(optimizer_config, model):
    pass
