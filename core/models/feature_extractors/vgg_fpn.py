import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict

# from avod import top_dir
# PRO_ROOT = top_dir()

# vgg_cfg = [3, 32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256]
vgg_cfg = [3, 64, 64, 'M', 128, 128, 'M', 512, 512, 512, 'M', 512, 512, 512]


class VGGFPN(nn.Module):
    def __init__(self, in_channels):
        super(VGGFPN, self).__init__()

        self.conv1 = self.__make_vgg_layers([32, 32], in_channels)
        self.conv2 = self.__make_vgg_layers(['M', 64, 64], 32)
        self.conv3 = self.__make_vgg_layers(['M', 128, 128, 128], 64)
        self.conv4 = self.__make_vgg_layers(['M', 256, 256, 256], 128)

        self.fusion3 = self.__make_fusion_layers(256, 64)
        self.fusion2 = self.__make_fusion_layers(128, 32)
        self.fusion1 = self.__make_fusion_layers(64, 32)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.in_channles = in_channels
        # self.conv1 = self.__make_vgg_layers([64, 64], in_channels)
        # self.conv2 = self.__make_vgg_layers(['M', 128, 128], 64)
        # self.conv3 = self.__make_vgg_layers(['M', 256, 256, 256], 128)
        # self.conv4 = self.__make_vgg_layers(['M', 512, 512, 512], 256)

        # self.fusion3 = self.__make_fusion_layers(512, 128)
        # self.fusion2 = self.__make_fusion_layers(256, 64)
        # self.fusion1 = self.__make_fusion_layers(128, 64)

        # self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.upconv2 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        # self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.upconv3(self.conv4(c3))

        p5 = self.fusion3(torch.cat((c3, c4), dim=1))
        p6 = self.fusion2(torch.cat((c2, self.upconv2(p5)), dim=1))
        p7 = self.fusion1(torch.cat((c1, self.upconv1(p6)), dim=1))

        return p7

    @staticmethod
    def __make_vgg_layers(cfg, in_channels, batch_norm=False):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)

    def __make_fusion_layers(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def load_pretrained_faster_rcnn_weights(self):
        if self.in_channles == 3:
            faster_rcnn_path = os.path.join(PRO_ROOT, 'weights/fov_faster_rcnn.pth')
        else:
            faster_rcnn_path = os.path.join(PRO_ROOT, 'weights/bev_faster_rcnn.pth')

        checkpoint = torch.load(faster_rcnn_path)
        org_state_dict = checkpoint['model']

        new_state_dict = OrderedDict()
        for k, v in org_state_dict.items():
            k_list = k.split('.')
            if k_list[0] == 'RCNN_base' and int(k_list[1]) <= 21:
                new_state_dict[k] = v

        converted_state_dict = OrderedDict()

        state_dict = self.state_dict()
        count = 0
        keys = new_state_dict.keys()
        for k, v in state_dict.items():
            if count < len(keys):
                converted_state_dict[k] = new_state_dict[keys[count]]
                count += 1
        state_dict.update(converted_state_dict)
        self.load_state_dict(state_dict)

if __name__ == '__main__':
    net = VGGFPN(in_channels=3).cuda()
    x = Variable(torch.rand((1, 3, 384, 1280))).cuda()
    bev_faster_rcnn = os.path.join(PRO_ROOT, 'weights/fov_faster_rcnn.pth')
    checkpoint = torch.load(bev_faster_rcnn)
    org_state_dict = checkpoint['model']

    new_state_dict = OrderedDict()
    for k, v in org_state_dict.items():
        k_list = k.split('.')
        if k_list[0] == 'RCNN_base' and int(k_list[1]) <= 21:
            new_state_dict[k] = v

    converted_state_dict = OrderedDict()

    state_dict = net.state_dict()
    count = 0
    keys = new_state_dict.keys()
    for k, v in state_dict.items():
        if count < len(keys):
            converted_state_dict[k] = new_state_dict[keys[count]]
            count += 1
    state_dict.update(converted_state_dict)
    net.load_state_dict(state_dict)
    y = net(x)
    print('y size is: ', y.cpu().data.size())
