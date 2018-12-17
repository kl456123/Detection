# -*- coding: utf-8 -*-
import torch


class BBox3DCoder(object):
    def __init__(self, coder_config):
        pass

    def decode_batch(self, deltas, boxes):
        """
        Args:
            deltas: shape(N,K*A,4)
            boxes: shape(N,K*A,4)
        """
        pass
        # if boxes.dim() == 3:
            # pass
        # elif boxes.dim() == 2:
            # boxes = boxes.expand_as(deltas)
        # else:
            # raise ValueError("The dimension of boxes should be 3 or 2")
        # widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
        # heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
        # ctr_x = boxes[:, :, 0] + 0.5 * widths
        # ctr_y = boxes[:, :, 1] + 0.5 * heights

        # dx = deltas[:, :, 0::4]
        # dy = deltas[:, :, 1::4]
        # dw = deltas[:, :, 2::4]
        # dh = deltas[:, :, 3::4]

        # pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
        # pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
        # pred_w = torch.exp(dw) * widths.unsqueeze(2)
        # pred_h = torch.exp(dh) * heights.unsqueeze(2)

        # pred_boxes = deltas.clone()
        # # x1
        # pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
        # # y1
        # pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
        # # x2
        # pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
        # # y2
        # pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

        # return pred_boxes

    # def encode_batch(self, bboxes, assigned_gt_boxes):
        # reg_targets_batch = self._encode_batch(bboxes, assigned_gt_boxes)

        # return reg_targets_batch

    def encode_batch(self, boxes_2d, coords):
        """
        Note that bbox_3d is just some points in image about 3d bbox
        Args:
            bbox_2d: shape(N,4)
            bbox_3d: shape(N,7)
        """
        center_x = (boxes_2d[:, 2] + boxes_2d[:, 0]) / 2
        center_y = (boxes_2d[:, 3] + boxes_2d[:, 1]) / 2
        center = torch.stack([center_x, center_y], dim=-1)
        w = (boxes_2d[:, 2] - boxes_2d[:, 0] + 1)
        h = (boxes_2d[:, 3] - boxes_2d[:, 1] + 1)
        dims = torch.stack([w, h], dim=-1)

        bbox_3d = coords[:, :-1].view(-1, 3, 2)
        bbox_3d = (bbox_3d - center.unsqueeze(1)) / dims.unsqueeze(1)
        y = (coords[:, -1:] - center[:, 1:]) / dims[:, 1:]
        coords = torch.cat([bbox_3d.view(-1, 6), y], dim=-1)

        # targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh),
        # 2)

        return coords
