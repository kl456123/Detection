# -*- coding: utf-8 -*-

from core.models.iou_faster_rcnn_model import IoUFasterRCNN
from core.models.faster_rcnn_model import FasterRCNN
# from core.models.two_rpn_model import TwoRPNModel
# from core.models.new_faster_rcnn_model import NewFasterRCNN
# from core.models.distance_faster_rcnn_model import DistanceFasterRCNN
# from core.models.refine_faster_rcnn_model import RefineFasterRCNN
# from core.models.gate_faster_rcnn_model import GateFasterRCNN
# from core.models.rfcn_model import RFCNModel
from core.models.semantic_faster_rcnn_model import SemanticFasterRCNN
from core.models.overlaps_faster_rcnn_model import OverlapsFasterRCNN
from core.models.semantic_both_faster_rcnn_model import SemanticBothFasterRCNN
from core.models.new_semantic_faster_rcnn_model import NewSemanticFasterRCNN
from core.models.LED_model import LEDFasterRCNN
from core.models.loss_faster_rcnn_model import LossFasterRCNN
from core.models.single_iou_model import SingleIoUFasterRCNN
from core.models.double_iou_faster_rcnn_model import DoubleIoUFasterRCNN
from core.models.double_iou_faster_rcnn_model_slow import SlowDoubleIoUFasterRCNN
from core.models.three_iou_faster_rcnn_model import ThreeIoUFasterRCNN
from core.models.three_iou_faster_rcnn_model_org import OrgThreeIoUFasterRCNN
from core.models.three_iou_faster_rcnn_model_org_ohem import OrgOHEMThreeIoUFasterRCNN
from core.models.cascade_faster_rcnn_model import CascadeFasterRCNN
from core.models.double_iou_faster_rcnn_model_second_stage import DoubleIoUSecondStageFasterRCNN
from core.models.three_iou_faster_rcnn_model_org_ohem_second import OrgOHEMThreeIoUSecondStageFasterRCNN
from core.models.fpn_faster_rcnn_model import FPNFasterRCNN
from core.models.detach_faster_rcnn_model import DetachFasterRCNN
from core.models.cls_gate_faster_rcnn_model import ClsGateFasterRCNN
from core.models.segment_faster_rcnn_model import SegmentFasterRCNN
from core.models.nosegment_faster_rcnn_model import NoSegmentFasterRCNN
from core.models.sinet_model import SINetModel
from core.models.semantic_sinet_model import SemanticSINetModel
from core.models.detach_sinet_model import DetachSINetModel
from core.models.detach_double_iou_faster_rcnn_model import DetachDoubleIOUFasterRCNN
from core.models.post_cls_faster_rcnn_model import PostCLSFasterRCNN
from core.models.post_iou_faster_rcnn_model import PostIOUFasterRCNN
from core.models.reg_model import RegFasterRCNN
# class ModelBuilder(object):
# def __init__(self, model_config):
# self.model_config = model_config

# def build(self):
# pass

# choose class or function


def build(model_config, training=True):
    net_arch = model_config['net']
    if net_arch == 'vgg16':
        fasterRCNN = vgg16(model_config)
    elif net_arch == 'resnet50':
        fasterRCNN = resnet(
            model_config,
            training, )
    elif net_arch == 'faster_rcnn':
        fasterRCNN = FasterRCNN(model_config)
    elif net_arch == 'fpn':
        fasterRCNN = FPNFasterRCNN(model_config)
    elif net_arch == 'two_rpn':
        fasterRCNN = TwoRPNModel(model_config)
    elif net_arch == 'new_faster_rcnn':
        fasterRCNN = NewFasterRCNN(model_config)
    elif net_arch == 'distance_faster_rcnn':
        fasterRCNN = DistanceFasterRCNN(model_config)
    elif net_arch == 'refine_faster_rcnn':
        fasterRCNN = RefineFasterRCNN(model_config)
    elif net_arch == 'gate_faster_rcnn':
        fasterRCNN = GateFasterRCNN(model_config)
    elif net_arch == 'iou_faster_rcnn':
        fasterRCNN = IoUFasterRCNN(model_config)
    elif net_arch == 'rfcn':
        fasterRCNN = RFCNModel(model_config)
    elif net_arch == 'semantic':
        fasterRCNN = SemanticFasterRCNN(model_config)
    elif net_arch == 'overlaps':
        fasterRCNN = OverlapsFasterRCNN(model_config)
    elif net_arch == 'semantic_both':
        fasterRCNN = SemanticBothFasterRCNN(model_config)
    elif net_arch == 'new_semantic':
        fasterRCNN = NewSemanticFasterRCNN(model_config)
    elif net_arch == 'LED':
        fasterRCNN = LEDFasterRCNN(model_config)
    elif net_arch == 'loss':
        fasterRCNN = LossFasterRCNN(model_config)
    elif net_arch == 'single_iou':
        fasterRCNN = SingleIoUFasterRCNN(model_config)
    elif net_arch == 'double_iou':
        fasterRCNN = DoubleIoUFasterRCNN(model_config)
    elif net_arch == 'double_iou_slow':
        fasterRCNN = SlowDoubleIoUFasterRCNN(model_config)
    elif net_arch == 'three_iou':
        fasterRCNN = ThreeIoUFasterRCNN(model_config)
    elif net_arch == 'three_iou_org':
        fasterRCNN = OrgThreeIoUFasterRCNN(model_config)
    elif net_arch == 'three_iou_org_ohem':
        fasterRCNN = OrgOHEMThreeIoUFasterRCNN(model_config)
    elif net_arch == 'cascade':
        fasterRCNN = CascadeFasterRCNN(model_config)
    elif net_arch == 'double_iou_second':
        fasterRCNN = DoubleIoUSecondStageFasterRCNN(model_config)
    elif net_arch == 'three_iou_org_ohem_second':
        fasterRCNN = OrgOHEMThreeIoUSecondStageFasterRCNN(model_config)
    elif net_arch == 'detach':
        fasterRCNN = DetachFasterRCNN(model_config)
    elif net_arch == 'cls_gate':
        fasterRCNN = ClsGateFasterRCNN(model_config)
    elif net_arch == 'segment':
        fasterRCNN = SegmentFasterRCNN(model_config)
    elif net_arch == 'no_segment':
        fasterRCNN = NoSegmentFasterRCNN(model_config)
    elif net_arch == 'sinet':
        fasterRCNN = SINetModel(model_config)
    elif net_arch == 'semantic_sinet':
        fasterRCNN = SemanticSINetModel(model_config)
    elif net_arch == 'detach_sinet':
        fasterRCNN = DetachSINetModel(model_config)
    elif net_arch == 'detach_double_iou':
        fasterRCNN = DetachDoubleIOUFasterRCNN(model_config)
    elif net_arch == 'post_cls':
        fasterRCNN = PostCLSFasterRCNN(model_config)
    elif net_arch == 'post_iou':
        fasterRCNN = PostIOUFasterRCNN(model_config)
    elif net_arch == 'reg':
        fasterRCNN = RegFasterRCNN(model_config)
    else:
        raise ValueError('net arch {} is not supported'.format(net_arch))

    if training:
        fasterRCNN.train()
    else:
        fasterRCNN.eval()
    # depercated
    # fasterRCNN.create_architecture()
    return fasterRCNN
