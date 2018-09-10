# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------




import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from model.faster_rcnn.faster_rcnn import _fasterRCNN


class vgg16(_fasterRCNN):
    def __init__(self, model_config):
        self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
        self.dout_base_model = 512
        self.pretrained = model_config['pretrained']
        self.class_agnostic = model_config['class_agnostic']
        self.img_channels = model_config['img_channels']

        _fasterRCNN.__init__(self, model_config)

    def _init_modules(self):
        if self.img_channels == 3:
            flag = False
        else:
            flag = True
        vgg = models.vgg16()
        # ipdb.set_trace()
        if self.pretrained:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict(
                {k: v
                 for k, v in list(state_dict.items()) if k in vgg.state_dict()})

        vgg.classifier = nn.Sequential(*list(
            vgg.classifier._modules.values())[:-1])

        # not using the last maxpool layer
        features = list(vgg.features._modules.values())
        if flag:
            features[0] = nn.Conv2d(
                self.img_channels, 64, kernel_size=3, stride=1, padding=1)
            self._first_layer = features[0]

            self.RCNN_base = nn.Sequential(*features[:-1])
        else:
            self.RCNN_base = nn.Sequential(*features[:-1])

        self.RCNN_top = vgg.classifier

        # not using the last maxpool layer
        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(4096, 4)
            # self.RCNN_ry_pred = nn.Linear(4096, 2)
        else:
            self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)
            # self.RCNN_ry_pred = nn.Linear(4096, 2 * self.n_classes)

    def _head_to_tail(self, pool5):

        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)

        return fc7
