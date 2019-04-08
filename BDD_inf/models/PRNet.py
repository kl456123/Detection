import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.nn import BatchNorm2d as bn
from BDD_inf.models.detection import Detection
from BDD_inf.models.backbone import ResNet, BasicBlock
from BDD_inf.models.detection import StridePoolBlock, TwoStageRetinaLayer


class PRNet(Detection):
    def __init__(self, model_cfg):
        super(PRNet, self).__init__()
        det_features = model_cfg['det_features']
        num_class = model_cfg['num_classes']

        self.features = ResNet(BasicBlock, [2, 2, 2, 2], 0.5)

        self.det_feature1 = StridePoolBlock(
            det_features, det_features, stride=1)
        self.det_feature2 = StridePoolBlock(det_features, det_features)
        self.det_feature3 = StridePoolBlock(det_features, det_features)
        self.det_feature4 = StridePoolBlock(det_features, det_features)
        self.det_feature5 = StridePoolBlock(det_features, det_features)
        self.det_feature6 = StridePoolBlock(det_features, det_features)

        self.multibox = TwoStageRetinaLayer(
            num_class, num_anchor=12, in_channels=det_features, num_bins=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, bn):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, _input):
        feature = self.features.forward(_input)
        det_feature = []
        feature = F.upsample(feature, scale_factor=2, mode='bilinear')

        feature = self.det_feature1(feature)
        det_feature.append(feature)

        feature = self.det_feature2(feature)
        det_feature.append(feature)

        feature = self.det_feature3(feature)
        det_feature.append(feature)

        feature = self.det_feature4(feature)
        det_feature.append(feature)

        feature = self.det_feature5(feature)
        det_feature.append(feature)

        feature = self.det_feature6(feature)
        det_feature.append(feature)

        loc1_preds, loc2_preds, os_preds, cls_preds, dims_3d_out, orients_out = self.multibox(
            det_feature)

        return loc1_preds, loc2_preds, os_preds, cls_preds, dims_3d_out, orients_out


class AffinityPropagate(nn.Module):
    def __init__(self, inters=5):
        super(AffinityPropagate, self).__init__()
        self.inters = inters

    def forward(self, guidance, feature_maps):
        # normalize features
        gate1 = self.normalize(guidance.narrow(1, 0, 5))
        gate2 = self.normalize(guidance.narrow(1, 5, 5))
        gate3 = self.normalize(guidance.narrow(1, 10, 5))
        gate4 = self.normalize(guidance.narrow(1, 15, 5))
        gate5 = self.normalize(guidance.narrow(1, 20, 5))
        gate6 = self.normalize(guidance.narrow(1, 25, 5))

        for i in range(self.inters):
            # one propagation
            feature_map_0 = self.propagation(gate1, feature_maps)
            feature_map_1 = self.propagation(gate2, feature_maps)
            feature_map_2 = self.propagation(gate3, feature_maps)
            feature_map_3 = self.propagation(gate4, feature_maps)
            feature_map_4 = self.propagation(gate5, feature_maps)
            feature_map_5 = self.propagation(gate6, feature_maps)

            feature_maps[0] = feature_map_0
            feature_maps[1] = feature_map_1
            feature_maps[2] = feature_map_2
            feature_maps[3] = feature_map_3
            feature_maps[4] = feature_map_4
            feature_maps[5] = feature_map_5

        return feature_maps

    def propagation(self, gate, feature_map):
        gate1 = gate.narrow(1, 0, 1)
        feature = gate1 * feature_map[0]

        for i in range(4):
            idx = i + 1
            gate1 = gate.narrow(1, idx, 1)
            feature = feature + gate1 * feature_map[idx]

        return feature

    @staticmethod
    def normalize(gate):
        """
        normalize to (-1, 1)
        :param gate:
        :return:
        """
        mean = torch.mean(gate, dim=1).unsqueeze(1)
        max_gate, _ = torch.max(gate, dim=1)
        min_gate, _ = torch.min(gate, dim=1)
        max_gate = max_gate.unsqueeze(1)
        min_gate = min_gate.unsqueeze(1)

        gate = (gate - mean) / (max_gate - min_gate + 0.0000001)
        return gate

    @staticmethod
    def wightning(gate):
        mean = torch.mean(gate, dim=1).unsqueeze(1)
        diff = gate - mean
        bias = torch.sqrt(torch.mean(torch.pow(diff, 2), dim=1).unsqueeze(1))
        return diff / bias
