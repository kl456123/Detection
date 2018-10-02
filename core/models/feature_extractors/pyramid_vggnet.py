# -*- coding: utf-8 -*-

from core.model import Model
import torchvision.models as models


class PyramidVggnet(Model):
    def forward(self):
        pass

    def init_param(self):
        pass

    def init_modules(self):
        vggnet = models.vgg16()

    def init_weights(self):
        pass
