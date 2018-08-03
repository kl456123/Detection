# -*- coding: utf-8 -*-

import torch.nn as nn


class Model(nn.Module):
    def __init__(self, model_config):
        super(Model, self).__init__()
        self.build(model_config)

    def build(self):
        pass

    def loss(self):
        pass
