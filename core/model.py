# -*- coding: utf-8 -*-

import torch.nn as nn


class Model(nn.Module):
    def __init__(self, model_config):
        super(Model, self).__init__()
        # store model_config
        self.init_param(model_config)

        # use config to set up model
        self.init_modules()

        # init weights
        self.init_weights()

        # freeze modules
        #  self.freeze_modules()

    def loss(self):
        pass

    def init_weights(self):
        pass

    def init_modules(self):
        pass

    def init_param(self, model_config):
        pass

    def freeze_modules(self):
        for param in self.parameters():
            param.requires_grad = False
