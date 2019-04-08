import math
import torch
import numpy
import itertools
import torch.nn.functional as F
# import matplotlib.pyplot as plt

from torch import nn
from BDD_inf.cfgs.train_cfgs.coco_train_cfg import TrainCFG
if TrainCFG['is_sync_bn']:
    from sync_batchnorm import SynchronizedBatchNorm2d as bn
else:
    from torch.nn import BatchNorm2d as bn


class Detection(nn.Module):
    def __init__(self):
        super(Detection, self).__init__()

    def forward(self, _input):
        pass

    def load_pretrained_weight(self, net):
        pass


class UnifiedRetinaLayer(nn.Module):
    def __init__(self, num_classes, num_anchor, in_channels=256):
        super(UnifiedRetinaLayer, self).__init__()
        self.num_anchors = num_anchor
        self.num_classes = num_classes

        self.loc_layers = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.num_anchors * 4,
                kernel_size=1))

        self.conf_layer = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.num_anchors * self.num_classes,
                kernel_size=1))

        self.iou_layer = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.num_anchors * 2,
                kernel_size=1))

    def forward(self, features):
        y_locs = []
        y_ious = []
        y_confs = []

        for i, x in enumerate(features):
            y_loc = self.loc_layers(x)
            N = y_loc.size(0)
            y_loc = y_loc.permute(0, 2, 3, 1).contiguous()
            y_loc = y_loc.view(N, -1, 4)
            y_locs.append(y_loc)

            y_iou = self.iou_layer(x)
            y_iou = y_iou.permute(0, 2, 3, 1).contiguous()
            y_iou = y_iou.view(N, -1, 2)
            y_ious.append(y_iou)

            y_conf = self.conf_layer(x)
            y_conf = y_conf.permute(0, 2, 3, 1).contiguous()
            y_conf = y_conf.view(N, -1, self.num_classes)
            y_confs.append(y_conf)

        loc_preds = torch.cat(y_locs, dim=1)
        iou_preds = torch.cat(y_ious, dim=1)
        conf_preds = torch.cat(y_confs, dim=1)

        return loc_preds, conf_preds, iou_preds


class TwoStageRetinaLayer(nn.Module):
    def __init__(self, num_classes, num_anchor, in_channels=256, num_bins=2):
        super(TwoStageRetinaLayer, self).__init__()
        self.num_anchors = num_anchor
        self.num_classes = num_classes

        self.loc_feature1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.loc_feature2 = nn.Sequential(
            nn.Conv2d(
                in_channels * 2, in_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.cls_feature1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.cls_feature2 = nn.Sequential(
            nn.Conv2d(
                in_channels * 2, in_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.os_out = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_anchors * 2,
            kernel_size=1)
        self.cls_out = nn.Conv2d(
            in_channels, self.num_anchors * self.num_classes, kernel_size=1)
        self.box_out1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_anchors * 4,
            kernel_size=1)
        self.box_out2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_anchors * 4,
            kernel_size=1)

        # add two branch to predict info of 3d bbox
        self.dims_3d_out = nn.Conv2d(
            in_channels=in_channels,
            out_channels=7 * 3 * self.num_anchors,  # 7 classes
            kernel_size=1)
        # use multibin here
        self.orients_out = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_bins * 4 * self.num_anchors,
            kernel_size=1)

        self.box_3d_feature = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

    def forward(self, features):
        y_locs1 = []
        y_locs2 = []
        y_os = []
        y_cls = []

        y_dims_3d = []
        y_orients = []

        for i, x in enumerate(features):
            # location out
            loc_feature = self.loc_feature1(x)
            loc1 = self.box_out1(loc_feature)

            N = loc1.size(0)
            loc1 = loc1.permute(0, 2, 3, 1).contiguous()
            loc1 = loc1.view(N, -1, 4)
            y_locs1.append(loc1)

            loc_feature = torch.cat([x, loc_feature], dim=1)
            loc_feature = self.loc_feature2(loc_feature)
            loc2 = self.box_out2(loc_feature)

            N = loc2.size(0)
            loc2 = loc2.permute(0, 2, 3, 1).contiguous()
            loc2 = loc2.view(N, -1, 4)
            loc2 += loc1
            y_locs2.append(loc2)

            # os out
            cls_feature = self.cls_feature1(x)
            os_out = self.os_out(cls_feature)
            os_out = os_out.permute(0, 2, 3, 1).contiguous()
            os_out = os_out.view(N, -1, 2)
            y_os.append(os_out)

            cls_feature = torch.cat([x, cls_feature], dim=1)
            cls_feature = self.cls_feature2(cls_feature)
            cls_out = self.cls_out(cls_feature)

            cls_out = cls_out.permute(0, 2, 3, 1).contiguous()
            cls_out = cls_out.view(N, -1, self.num_classes)
            y_cls.append(cls_out)

            # box_3d_feature
            box_3d_feature = self.box_3d_feature(x)
            dims_3d_out = self.dims_3d_out(box_3d_feature)
            dims_3d_out = dims_3d_out.permute(0, 2, 3, 1).contiguous()
            dims_3d_out = dims_3d_out.view(N, -1, dims_3d_out.shape[-1])
            y_dims_3d.append(dims_3d_out)

            # orient
            orients_out = self.orients_out(box_3d_feature)
            orients_out = orients_out.permute(0, 2, 3, 1).contiguous()
            orients_out = orients_out.view(N, -1, orients_out.shape[-1])
            y_orients.append(orients_out)

        loc1_preds = torch.cat(y_locs1, dim=1)
        loc2_preds = torch.cat(y_locs2, dim=1)
        os_preds = torch.cat(y_os, dim=1)
        cls_preds = torch.cat(y_cls, dim=1)

        y_orients = torch.cat(y_orients, dim=1)
        y_dims_3d = torch.cat(y_dims_3d, dim=1)

        return loc1_preds, loc2_preds, os_preds, cls_preds, y_dims_3d, y_orients


class LightTwoStageRetinaLayer(nn.Module):
    def __init__(self, num_classes, num_anchor, in_channels=256):
        super(LightTwoStageRetinaLayer, self).__init__()
        self.num_anchors = num_anchor
        self.num_classes = num_classes

        self.loc_feature1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.loc_feature2 = nn.Sequential(
            nn.Conv2d(
                in_channels * 2, in_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.cls_feature1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.cls_feature2 = nn.Sequential(
            nn.Conv2d(
                in_channels * 2, in_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.os_out = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_anchors * 2,
            kernel_size=1)
        self.cls_out = nn.Conv2d(
            in_channels, self.num_anchors * self.num_classes, kernel_size=1)
        self.box_out1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_anchors * 4,
            kernel_size=1)
        self.box_out2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_anchors * 4,
            kernel_size=1)

    def forward(self, features):
        y_locs1 = []
        y_locs2 = []
        y_os = []
        y_cls = []

        for i, x in enumerate(features):
            # location out
            loc_feature = self.loc_feature1(x)
            loc1 = self.box_out1(loc_feature)

            N = loc1.size(0)
            loc1 = loc1.permute(0, 2, 3, 1).contiguous()
            loc1 = loc1.view(N, -1, 4)
            y_locs1.append(loc1)

            loc_feature = torch.cat([x, loc_feature], dim=1)
            loc_feature = self.loc_feature2(loc_feature)
            loc2 = self.box_out2(loc_feature)

            N = loc2.size(0)
            loc2 = loc2.permute(0, 2, 3, 1).contiguous()
            loc2 = loc2.view(N, -1, 4)
            loc2 += loc1
            y_locs2.append(loc2)

            # os out
            cls_feature = self.cls_feature1(x)
            os_out = self.os_out(cls_feature)
            os_out = os_out.permute(0, 2, 3, 1).contiguous()
            os_out = os_out.view(N, -1, 2)
            y_os.append(os_out)

            cls_feature = torch.cat([x, cls_feature], dim=1)
            cls_feature = self.cls_feature2(cls_feature)
            cls_out = self.cls_out(cls_feature)

            cls_out = cls_out.permute(0, 2, 3, 1).contiguous()
            cls_out = cls_out.view(N, -1, self.num_classes)
            y_cls.append(cls_out)

        loc1_preds = torch.cat(y_locs1, dim=1)
        loc2_preds = torch.cat(y_locs2, dim=1)
        os_preds = torch.cat(y_os, dim=1)
        cls_preds = torch.cat(y_cls, dim=1)

        return loc1_preds, loc2_preds, os_preds, cls_preds


class ScaleFusingRetinaLayer(nn.Module):
    def __init__(self, num_classes, num_anchor, in_channels=256):
        super(ScaleFusingRetinaLayer, self).__init__()
        self.num_anchors = num_anchor
        self.num_classes = num_classes

        self.loc_feature1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.cls_feature1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.loc_feature2 = nn.Sequential(
            nn.Conv2d(
                in_channels * 2, in_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.cls_feature2 = nn.Sequential(
            nn.Conv2d(
                in_channels * 2, in_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.os_out = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_anchors * 2,
            kernel_size=1)
        self.cls_out = nn.Conv2d(
            in_channels, self.num_anchors * self.num_classes, kernel_size=1)
        self.box_out1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_anchors * 4,
            kernel_size=1)
        self.box_out2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_anchors * 4,
            kernel_size=1)

    def forward(self, ori_feature, fuse_feature):
        y_locs1 = []
        y_locs2 = []
        y_os = []
        y_cls = []

        for x_ori, x_fuse in zip(ori_feature, fuse_feature):
            # location out
            loc_feature = self.loc_feature1(x_ori)
            loc1 = self.box_out1(loc_feature)

            N = loc1.size(0)
            loc1 = loc1.permute(0, 2, 3, 1).contiguous()
            loc1 = loc1.view(N, -1, 4)
            y_locs1.append(loc1)

            loc_feature = torch.cat([x_fuse, loc_feature], dim=1)
            loc_feature = self.loc_feature2(loc_feature)
            loc2 = self.box_out2(loc_feature)

            N = loc2.size(0)
            loc2 = loc2.permute(0, 2, 3, 1).contiguous()
            loc2 = loc2.view(N, -1, 4)
            loc2 += loc1
            y_locs2.append(loc2)

            # os out
            cls_feature = self.cls_feature1(x_ori)
            os_out = self.os_out(cls_feature)
            os_out = os_out.permute(0, 2, 3, 1).contiguous()
            os_out = os_out.view(N, -1, 2)
            y_os.append(os_out)

            cls_feature = torch.cat([x_fuse, cls_feature], dim=1)
            cls_feature = self.cls_feature2(cls_feature)
            cls_out = self.cls_out(cls_feature)

            cls_out = cls_out.permute(0, 2, 3, 1).contiguous()
            cls_out = cls_out.view(N, -1, self.num_classes)
            y_cls.append(cls_out)

        loc1_preds = torch.cat(y_locs1, dim=1)
        loc2_preds = torch.cat(y_locs2, dim=1)
        os_preds = torch.cat(y_os, dim=1)
        cls_preds = torch.cat(y_cls, dim=1)

        return loc1_preds, loc2_preds, os_preds, cls_preds


class TwoStageROIRetinaLayer(nn.Module):
    def __init__(self, num_classes, num_anchor, in_channels=256):
        super(TwoStageROIRetinaLayer, self).__init__()
        self.num_anchors = num_anchor
        self.num_classes = num_classes

        self.loc_feature1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.cls_feature1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.roi_attention = nn.Sequential(
            nn.Conv2d(
                in_channels * 3, in_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                in_channels, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(inplace=True))

        self.cls_roi = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True))

        self.box_roi = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True))

        self.loc_res = nn.Sequential(
            nn.Conv2d(
                in_channels * 2, in_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.num_anchors * 4,
                kernel_size=1))

        self.cls_out = nn.Sequential(
            nn.Conv2d(
                in_channels * 2, in_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                self.num_anchors * self.num_classes,
                kernel_size=1))

        self.os_out = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_anchors * 2,
            kernel_size=1)
        self.box_out1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_anchors * 4,
            kernel_size=1)

    def forward(self, features, ori_feature):
        y_locs1 = []
        y_locs2 = []
        y_os = []
        y_cls = []
        ori_size = ori_feature.size()

        for x in features:
            # location out1
            loc_feature = self.loc_feature1(x)
            loc1 = self.box_out1(loc_feature)
            N = loc1.size(0)

            loc1 = loc1.permute(0, 2, 3, 1).contiguous()
            loc1 = loc1.view(N, -1, 4)
            y_locs1.append(loc1)

            # os out
            cls_feature = self.cls_feature1(x)
            os_out = self.os_out(cls_feature)

            os_out = os_out.permute(0, 2, 3, 1).contiguous()
            os_out = os_out.view(N, -1, 2)
            y_os.append(os_out)

            # ROI mask
            roi_mask = self.roi_attention(
                torch.cat([x, loc_feature, cls_feature], dim=1))

            mask_size = roi_mask.size()
            roi_mask = F.interpolate(
                roi_mask, size=(ori_size[2], ori_size[3]), mode='bilinear')
            # uncomment to visualize mask
            # roi_mask[roi_mask < 0.2] = 0
            # roi_mask = norm_2d(roi_mask)
            # plt.imsave('heatmap{}.png'.format(i), roi_mask.squeeze().squeeze().to('cpu').numpy())
            roi_feature = torch.mul(roi_mask, ori_feature)
            cls_roi = self.cls_roi(roi_feature)
            box_roi = self.box_roi(roi_feature)

            cls_roi = F.interpolate(
                cls_roi, size=(mask_size[2], mask_size[3]), mode='bilinear')
            box_roi = F.interpolate(
                box_roi, size=(mask_size[2], mask_size[3]), mode='bilinear')

            # location out2
            loc2 = self.loc_res(torch.cat([loc_feature, box_roi], dim=1))
            loc2 = loc2.permute(0, 2, 3, 1).contiguous()
            loc2 = loc2.view(N, -1, 4)

            loc2 += loc1
            y_locs2.append(loc2)

            # cls out2
            cls_out = self.cls_out(torch.cat([cls_feature, cls_roi], dim=1))
            cls_out = cls_out.permute(0, 2, 3, 1).contiguous()
            cls_out = cls_out.view(N, -1, self.num_classes)
            y_cls.append(cls_out)

        loc1_preds = torch.cat(y_locs1, dim=1)
        loc2_preds = torch.cat(y_locs2, dim=1)
        os_preds = torch.cat(y_os, dim=1)
        cls_preds = torch.cat(y_cls, dim=1)

        return loc1_preds, loc2_preds, os_preds, cls_preds


def norm_2d(x):
    x -= x.min()
    x /= x.max()
    return x


class ROIAttentionRetinaLayer(nn.Module):
    def __init__(self, num_classes, num_anchor, in_channels=256):
        super(ROIAttentionRetinaLayer, self).__init__()
        self.num_anchors = num_anchor
        self.num_classes = num_classes

        self.loc_feature1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.cls_feature1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.roi_attention = nn.Sequential(
            nn.Conv2d(
                in_channels * 3, in_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1),
            bn(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=1,
                kernel_size=3,
                stride=1,
                padding=1),
            bn(num_features=1),
            nn.ReLU(inplace=True), )

        self.loc_feature2 = nn.Sequential(
            nn.Conv2d(
                in_channels * 3, in_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.cls_feature2 = nn.Sequential(
            nn.Conv2d(
                in_channels * 3, in_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True), )

        self.os_out = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_anchors * 2,
            kernel_size=1)
        self.cls_out = nn.Conv2d(
            in_channels, self.num_anchors * self.num_classes, kernel_size=1)
        self.box_out1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_anchors * 4,
            kernel_size=1)
        self.box_out2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_anchors * 4,
            kernel_size=1)

    def forward(self, features):
        y_locs1 = []
        y_locs2 = []
        y_os = []
        y_cls = []

        for i, x in enumerate(features):
            # location out1
            loc_feature = self.loc_feature1(x)
            loc1 = self.box_out1(loc_feature)
            N = loc1.size(0)

            loc1 = loc1.permute(0, 2, 3, 1).contiguous()
            loc1 = loc1.view(N, -1, 4)
            y_locs1.append(loc1)

            # os out
            cls_feature = self.cls_feature1(x)
            os_out = self.os_out(cls_feature)

            os_out = os_out.permute(0, 2, 3, 1).contiguous()
            os_out = os_out.view(N, -1, 2)
            y_os.append(os_out)

            # ROI Attention feature
            total_feature = torch.cat([x, loc_feature, cls_feature], dim=1)
            roi_attention = self.roi_attention(total_feature)

            # location out2
            loc_feature = self.loc_feature2(total_feature)
            loc_feature = loc_feature + torch.mul(loc_feature, roi_attention)
            loc2 = self.box_out2(loc_feature)
            loc2 = loc2.permute(0, 2, 3, 1).contiguous()
            loc2 = loc2.view(N, -1, 4)

            loc2 += loc1
            y_locs2.append(loc2)

            # cls out2
            cls_feature = self.cls_feature2(total_feature)
            cls_feature = cls_feature + torch.mul(cls_feature, roi_attention)
            cls_out = self.cls_out(cls_feature)

            cls_out = cls_out.permute(0, 2, 3, 1).contiguous()
            cls_out = cls_out.view(N, -1, self.num_classes)
            y_cls.append(cls_out)

        loc1_preds = torch.cat(y_locs1, dim=1)
        loc2_preds = torch.cat(y_locs2, dim=1)
        os_preds = torch.cat(y_os, dim=1)
        cls_preds = torch.cat(y_cls, dim=1)

        return loc1_preds, loc2_preds, os_preds, cls_preds


class RetinaPriorBox(object):
    """
        * Compute priorbox coordinates in center-offset form for each source feature map.
    """

    def __init__(self):
        super(RetinaPriorBox, self).__init__()

    def __call__(self, cfg):
        self.image_size = cfg['input_shape']
        self.aspect_ratios = cfg['aspect_ratio']
        self.default_ratio = cfg['default_ratio']
        self.output_stride = cfg['output_scale']
        self.clip = True

        scale_w = self.image_size[0]
        scale_h = self.image_size[1]
        steps_w = [s / scale_w for s in self.output_stride]
        steps_h = [s / scale_h for s in self.output_stride]
        sizes = self.default_ratio
        aspect_ratios = self.aspect_ratios

        feature_map_w = [
            int(math.floor(scale_w / s)) for s in self.output_stride
        ]
        feature_map_h = [
            int(math.floor(scale_h / s)) for s in self.output_stride
        ]
        assert len(feature_map_h) == len(feature_map_w)
        num_layers = len(feature_map_h)

        boxes = []
        for i in range(num_layers):
            fm_w = feature_map_w[i]
            fm_h = feature_map_h[i]
            for h, w in itertools.product(range(fm_h), range(fm_w)):
                cx = (w + 0.5) * steps_w[i]
                cy = (h + 0.5) * steps_h[i]

                s = sizes[i]
                for ar in aspect_ratios[i]:
                    boxes.append(
                        (cx, cy, s * math.sqrt(ar), 2 * s / math.sqrt(ar)))
                    boxes.append(
                        (cx, cy, s / math.sqrt(ar), 2 * s * math.sqrt(ar)))

                s = sizes[i] * math.pow(2, 1. / 3)
                for ar in aspect_ratios[i]:
                    boxes.append(
                        (cx, cy, s * math.sqrt(ar), 2 * s / math.sqrt(ar)))
                    boxes.append(
                        (cx, cy, s / math.sqrt(ar), 2 * s * math.sqrt(ar)))

                s = sizes[i] * math.pow(2, 2. / 3)
                for ar in aspect_ratios[i]:
                    boxes.append(
                        (cx, cy, s * math.sqrt(ar), 2 * s / math.sqrt(ar)))
                    boxes.append(
                        (cx, cy, s / math.sqrt(ar), 2 * s * math.sqrt(ar)))

        boxes = numpy.array(boxes, dtype=float)
        boxes = torch.from_numpy(boxes).float()  # back to torch land
        if self.clip:
            boxes.clamp_(min=0., max=1.)
        return boxes


class StridePoolBlock(nn.Sequential):
    def __init__(self, input_num, out_num, stride=2):
        super(StridePoolBlock, self).__init__()
        self.add_module(
            'conv3',
            nn.Conv2d(
                in_channels=input_num,
                out_channels=out_num,
                kernel_size=3,
                stride=stride,
                padding=1))
        self.add_module('bn', bn(num_features=out_num))
        self.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, _input):
        _feature = super(StridePoolBlock, self).forward(_input)

        return _feature


class ScaleFusingBlock(nn.Sequential):
    def __init__(self, input_num, out_num):
        super(ScaleFusingBlock, self).__init__()
        self.add_module(
            'conv',
            nn.Conv2d(
                in_channels=input_num,
                out_channels=out_num,
                kernel_size=1,
                bias=False))

    def forward(self, _input):
        _feature = super(ScaleFusingBlock, self).forward(_input)

        return _feature


class ScalePropagationBlock(nn.Sequential):
    def __init__(self, input_num, out_num):
        super(ScalePropagationBlock, self).__init__()
        self.add_module(
            'conv0',
            nn.Conv2d(
                in_channels=input_num, out_channels=out_num, kernel_size=1))
        self.add_module('relu0', nn.ReLU(inplace=True))
        self.add_module(
            'conv1',
            nn.Conv2d(
                in_channels=out_num,
                out_channels=out_num,
                kernel_size=3,
                padding=1))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module(
            'conv2',
            nn.Conv2d(
                in_channels=out_num,
                out_channels=out_num,
                kernel_size=3,
                padding=1))
        self.add_module('relu2', nn.ReLU(inplace=True))

    def forward(self, _input, out_size=(80, 80)):
        _feature = super(ScalePropagationBlock, self).forward(_input)

        return _feature


class DenseAsppLayer(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self,
                 input_num,
                 num1,
                 num2,
                 dilation_rate,
                 drop_out,
                 bn_start=True):
        super(DenseAsppLayer, self).__init__()
        if bn_start:
            self.add_module('norm1', bn(num_features=input_num)),

        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module(
            'conv1',
            nn.Conv2d(
                in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm2', bn(num_features=num1)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module(
            'conv2',
            nn.Conv2d(
                in_channels=num1,
                out_channels=num2,
                kernel_size=3,
                dilation=dilation_rate,
                padding=dilation_rate)),

        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(DenseAsppLayer, self).forward(_input)

        if self.drop_rate > 0:
            feature = F.dropout(
                feature, p=self.drop_rate, training=self.training)

        return torch.cat([_input, feature], 1)


class DenseAsppBlock(nn.Module):
    def __init__(self, in_channel):
        super(DenseAsppBlock, self).__init__()
        num_features = in_channel
        self.ASPP_1 = DenseAsppLayer(
            input_num=num_features,
            num1=64,
            num2=32,
            dilation_rate=1,
            drop_out=0)
        num_features += 32
        self.ASPP_2 = DenseAsppLayer(
            input_num=num_features,
            num1=64,
            num2=32,
            dilation_rate=2,
            drop_out=0)
        num_features += 32
        self.ASPP_3 = DenseAsppLayer(
            input_num=num_features,
            num1=64,
            num2=32,
            dilation_rate=3,
            drop_out=0)
        num_features += 32

    def forward(self, _input):
        _input = self.ASPP_1.forward(_input)
        _input = self.ASPP_2.forward(_input)
        _input = self.ASPP_3.forward(_input)

        return _input
