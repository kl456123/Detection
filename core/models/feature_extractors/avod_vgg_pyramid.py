# -*- coding: utf-8 -*-

from core.model import Model
import torch.nn as nn
import torch


class AVODVGGPyramidExtractor(Model):
    def init_param(self, model_config):
        self.input_mode = model_config['input_mode']
        self.l2_weights_decay = model_config['l2_weights_decay']
        self.ndin = model_config['ndin']

    def init_weights(self):
        pass

    def init_modules(self):
        # padding
        if self.input_mode == 'bev':
            self.padding = nn.ConstantPad2d((0, 0, 4, 0), 0)
        # encode
        self.conv1_1 = nn.Conv2d(self.ndin, 32, 3, 1, 1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.relu1_1 = nn.ReLU()

        self.conv1_2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.relu1_2 = nn.ReLU()

        self.pool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.relu2_1 = nn.ReLU()

        self.conv2_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.relu2_2 = nn.ReLU()

        self.pool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3_1 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu3_1 = nn.ReLU()

        self.conv3_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.relu3_2 = nn.ReLU()

        self.conv3_3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn3_3 = nn.BatchNorm2d(128)
        self.relu3_3 = nn.ReLU()

        self.pool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.relu4_1 = nn.ReLU()

        self.conv4_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.relu4_2 = nn.ReLU()

        self.conv4_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn4_3 = nn.BatchNorm2d(256)
        self.relu4_3 = nn.ReLU()

        # decode
        # upsample and fusion
        self.upconv3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bn5_1 = nn.BatchNorm2d(128)
        self.relu5_1 = nn.ReLU()

        # fusion layer
        self.pyramid_fusion3 = nn.Conv2d(256, 64, 3, 1, 1)
        self.bn6_1 = nn.BatchNorm2d(64)
        self.relu6_1 = nn.ReLU()

        self.upconv2 = nn.ConvTranspose2d(64, 64, 4, 2, 1)
        self.bn7_1 = nn.BatchNorm2d(64)
        self.relu7_1 = nn.ReLU()

        self.pyramid_fusion2 = nn.Conv2d(128, 32, 3, 1, 1)
        self.bn8_1 = nn.BatchNorm2d(32)
        self.relu8_1 = nn.ReLU()

        self.upconv1 = nn.ConvTranspose2d(32, 32, 4, 2, 1)
        self.bn9_1 = nn.BatchNorm2d(32)
        self.relu9_1 = nn.ReLU()

        self.pyramid_fusion1 = nn.Conv2d(64, 32, 3, 1, 1)
        self.bn10_1 = nn.BatchNorm2d(32)
        self.relu10_1 = nn.ReLU()

    def forward(self, input_image):
        if self.input_mode == 'bev':
            # pad input
            padded = self.padding(input_image)
        else:
            padded = input_image

        # encode
        net = self.conv1_1(padded)
        net = self.bn1_1(net)
        net = self.relu1_1(net)

        net = self.conv1_2(net)
        net = self.bn1_2(net)
        conv1 = self.relu1_2(net)

        net = self.pool1(conv1)

        net = self.conv2_1(net)
        net = self.bn2_1(net)
        net = self.relu2_1(net)

        net = self.conv2_2(net)
        net = self.bn2_2(net)
        conv2 = self.relu2_2(net)

        net = self.pool2(conv2)

        net = self.conv3_1(net)
        net = self.bn3_1(net)
        net = self.relu3_1(net)

        net = self.conv3_2(net)
        net = self.bn3_2(net)
        net = self.relu3_2(net)

        net = self.conv3_3(net)
        net = self.bn3_3(net)
        conv3 = self.relu3_3(net)

        net = self.pool3(conv3)

        net = self.conv4_1(net)
        net = self.bn4_1(net)
        net = self.relu4_1(net)

        net = self.conv4_2(net)
        net = self.bn4_2(net)
        net = self.relu4_2(net)

        net = self.conv4_3(net)
        net = self.bn4_3(net)
        conv4 = self.relu4_3(net)

        # decoder
        upconv3 = self.upconv3(conv4)
        concat3 = torch.cat([conv3, upconv3],dim=1)
        pyramid_fusion3 = self.pyramid_fusion3(concat3)

        upconv2 = self.upconv2(pyramid_fusion3)
        concat2 = torch.cat([conv2, upconv2],dim=1)
        pyramid_fusion2 = self.pyramid_fusion2(concat2)

        upconv1 = self.upconv1(pyramid_fusion2)
        concat1 = torch.cat([conv1, upconv1],dim=1)
        pyramid_fusion1 = self.pyramid_fusion1(concat1)

        if self.input_mode == 'bev':
            sliced = pyramid_fusion1[:, :, 4:, :]
        else:
            sliced = pyramid_fusion1

        return sliced
