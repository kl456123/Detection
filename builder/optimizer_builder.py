# -*- coding: utf-8 -*-

import torch


class OptimizerBuilder(object):
    def __init__(self, module, optim_config):
        self.module = module
        self.optim_config = optim_config

    def get_params(self):
        """
        This method can be overload
        """

        #  return self.module.parameters()
        all_params = self.module.parameters()
        learnable_params = [
            param for param in all_params if param.requires_grad
        ]
        return learnable_params

    def build(self):
        cfg = self.optim_config
        params = self.get_params()

        if cfg['type'] == "adam":
            optimizer = torch.optim.Adam(params)

        elif cfg['type'] == "sgd":
            optimizer = torch.optim.SGD(params,
                                        momentum=cfg['momentum'],
                                        lr=cfg['lr'])
        else:
            raise ValueError(
                'this type of optimizer is unknown, please change it!')
        return optimizer
