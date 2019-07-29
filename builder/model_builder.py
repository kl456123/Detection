# -*- coding: utf-8 -*-

# from core.models.iou_faster_rcnn_model import IoUFasterRCNN
# from core.models.faster_rcnn_model import FasterRCNN
# from core.models.two_rpn_model import TwoRPNModel
# from core.models.new_faster_rcnn_model import NewFasterRCNN
# from core.models.distance_faster_rcnn_model import DistanceFasterRCNN
# from core.models.refine_faster_rcnn_model import RefineFasterRCNN
# from core.models.gate_faster_rcnn_model import GateFasterRCNN
# from core.models.rfcn_model import RFCNModel
# from core.models.semantic_faster_rcnn_model import SemanticFasterRCNN
# from core.models.overlaps_faster_rcnn_model import OverlapsFasterRCNN
# from core.models.semantic_both_faster_rcnn_model import SemanticBothFasterRCNN
# from core.models.new_semantic_faster_rcnn_model import NewSemanticFasterRCNN
# from core.models.LED_model import LEDFasterRCNN
# from core.models.loss_faster_rcnn_model import LossFasterRCNN
# from core.models.single_iou_model import SingleIoUFasterRCNN
# from core.models.double_iou_faster_rcnn_model import DoubleIoUFasterRCNN
# from core.models.double_iou_faster_rcnn_model_slow import SlowDoubleIoUFasterRCNN
# from core.models.three_iou_faster_rcnn_model import ThreeIoUFasterRCNN
# from core.models.three_iou_faster_rcnn_model_org import OrgThreeIoUFasterRCNN
# from core.models.three_iou_faster_rcnn_model_org_ohem import OrgOHEMThreeIoUFasterRCNN
# from core.models.cascade_faster_rcnn_model import CascadeFasterRCNN
# from core.models.double_iou_faster_rcnn_model_second_stage import DoubleIoUSecondStageFasterRCNN
# from core.models.three_iou_faster_rcnn_model_org_ohem_second import OrgOHEMThreeIoUSecondStageFasterRCNN
# from core.models.fpn_faster_rcnn_model import FPNFasterRCNN
# from core.models.mono_3d import Mono3DFasterRCNN
#  from core.models.mono_3d_angle import Mono3DAngleFasterRCNN
# from core.models.mono_3d_angle_simpler import Mono3DAngleSimplerFasterRCNN
# from core.models.mono_3d_simpler import Mono3DSimplerFasterRCNN
# from core.models.mono_3d_angle_new import Mono3DAngleNewFasterRCNN
# from core.models.ssd_model import SSDModel
# from core.models.oft_model import OFTModel
# from core.models.mono_3d_better import Mono3DBetterFasterRCNN
# from core.models.refine_oft_model import RefineOFTModel
# from core.models.oft_4c_model import OFT4CModel
# from core.models.mono_3d_final import Mono3DFinalFasterRCNN
#  from core.models.mono_3d_final_plus import Mono3DFinalPlusFasterRCNN
from core.models.geometry_v3 import GeometryV3FasterRCNN
from core.models.geometry_v2 import GeometryV2FasterRCNN
from core.models.geometry_v1 import GeometryV1FasterRCNN


def build(model_config, training=True):
    net_arch = model_config['net']
    if net_arch == 'geometry_v1':
        fasterRCNN = GeometryV1FasterRCNN(model_config)
    elif net_arch == 'geometry_v2':
        fasterRCNN = GeometryV2FasterRCNN(model_config)
    elif net_arch == 'geometry_v3':
        fasterRCNN = GeometryV3FasterRCNN(model_config)
    else:
        raise ValueError('net arch {} is not supported'.format(net_arch))

    if training:
        fasterRCNN.train()
    else:
        fasterRCNN.eval()

    fasterRCNN.pre_forward()
    # depercated
    # fasterRCNN.create_architecture()
    return fasterRCNN
