import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta


class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, model_config):
        super(_fasterRCNN, self).__init__()
        classes = model_config['classes']
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = model_config['class_agnostic']
        self.pooling_size = model_config['pooling_size']
        self.pooling_mode = model_config['pooling_mode']
        self.crop_resize_with_max_pool = model_config[
            'crop_resize_with_max_pool']
        self.truncated = model_config['truncated']

        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(model_config['rpn_config'])
        self.RCNN_proposal_target = _ProposalTargetLayer(
            model_config['proposal_target_layer_config'])
        self.RCNN_roi_pool = _RoIPooling(self.pooling_size, self.pooling_size,
                                         1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg(self.pooling_size, self.pooling_size,
                                          1.0 / 16.0)

        self.grid_size = self.pooling_size * 2 if self.crop_resize_with_max_pool else self.pooling_size
        self.RCNN_roi_crop = _RoICrop()
        self.l2loss = nn.MSELoss(reduce=False)

    def forward(self, im_data, im_info, gt_boxes, num_boxes, gt_ry=None):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        # gt_ry = gt_ry.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info,
                                                          gt_boxes, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(
                rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(
                rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if self.pooling_mode == 'crop':
            grid_xy = _affine_grid_gen(
                rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack(
                [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]],
                3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat,
                                             Variable(grid_yx).detach())
            if self.crop_resize_with_max_pool:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif self.pooling_mode == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
        elif self.pooling_mode == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1, 5))

        # feed pooled features to top model
        pooled_feat = self._head_to_tail(pooled_feat)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(
                bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(
                bbox_pred_view, 1,
                rois_label.view(rois_label.size(0), 1, 1).expand(
                    rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        # compute ry
        # ry_pred = self.RCNN_ry_pred(pooled_feat)

        # compute object classification probability
        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob = F.softmax(cls_score)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target,
                                             rois_inside_ws, rois_outside_ws)

            # ry regression L2 loss
            # import ipdb
            # ipdb.set_trace()
            # RCNN_loss_ry = self.l2loss(ry_pred, rois_ry_target)
            # RCNN_loss_ry = 0

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)

        predict = {

            # pred
            'rois': rois,
            'rois_label': rois_label,
            'cls_prob': cls_prob,
            'bbox_pred': bbox_pred,

            # loss
            'rpn_loss_cls': rpn_loss_cls,
            'rpn_loss_bbox': rpn_loss_bbox,
            'RCNN_loss_cls': RCNN_loss_cls,
            'RCNN_loss_bbox': RCNN_loss_bbox,
        }

        # return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label
        return predict

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(
                    mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        if hasattr(self, '_first_layer'):
            normal_init(self._first_layer, 0, 0.001, self.truncated)
        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, self.truncated)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, self.truncated)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, self.truncated)
        normal_init(self.RCNN_cls_score, 0, 0.01, self.truncated)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, self.truncated)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    # def get_params(self, train_config):
    # params = []
    # lr = train_config['lr']
    # for key, value in dict(self.named_parameters()).items():
    # if value.requires_grad:
    # if 'bias' in key:
    # params += [{
    # 'params': [value],
    # 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
    # 'weight_decay': cfg.TRAIN.BIAS_DECAY and
    # cfg.TRAIN.WEIGHT_DECAY or 0
    # }]
    # else:
    # params += [{
    # 'params': [value],
    # 'lr': lr,
    # 'weight_decay': cfg.TRAIN.WEIGHT_DECAY
    # }]
    # return params
