# -*- coding: utf-8 -*-
import torch
import math
from core.ops import get_angle
from torch.nn import functional as F

from utils import geometry_utils
from core.utils import tensor_utils
from utils import encoder_utils


class BBox3DCoder(object):
    def __init__(self, coder_config):
        self.mean_dims = None

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

    def encode_batch_dims(self, boxes_2d, dims):
        """
        encoding dims may be better,here just encode dims_2d
        Args:
            boxes_2d: shape(N,)
            dims: shape(N,6), (h,w,l) and their projection in 2d
        """
        w = (boxes_2d[:, 2] - boxes_2d[:, 0] + 1)
        h = (boxes_2d[:, 3] - boxes_2d[:, 1] + 1)

        target_h = torch.log(dims[:, 0] / h)
        target_w = torch.log(dims[:, 1] / w)
        target_l = torch.log(dims[:, 2] / w)

        h_3d_mean = 1.67
        w_3d_mean = 1.87
        l_3d_mean = 3.7

        h_3d_std = 1
        w_3d_std = 1
        l_3d_std = 1

        target_h_3d = (dims[:, 3] - h_3d_mean) / h_3d_std
        target_w_3d = (dims[:, 4] - w_3d_mean) / w_3d_std
        target_l_3d = (dims[:, 5] - l_3d_mean) / l_3d_std
        targets = torch.stack(
            [
                target_h, target_w, target_l, target_h_3d, target_w_3d,
                target_l_3d
            ],
            dim=-1)
        return targets

    def reorder_lines(self, lines):

        # sort by x
        _, order = torch.sort(lines[..., 0], dim=-1)

        lines = tensor_utils.multidim_index(lines, order)
        return lines

    def find_nearest_between_two_lines(self, lines, lines_2d):
        lines = lines.view(-1, 2, 2, 3)
        lines_2d = lines_2d.view(-1, 2, 2, 2)
        mid_points = lines.mean(dim=2)  # (N, 2, 3)
        dist = mid_points.norm(dim=-1)

        _, visible_index = torch.min(dist, dim=-1)
        row = torch.arange(visible_index.numel()).type_as(visible_index)
        # may be one of them or may be none of them
        near_side = lines_2d[row, visible_index]

        # import ipdb
        # ipdb.set_trace()
        # calc visible
        left_slope = geometry_utils.torch_line_to_orientation(
            lines_2d[:, 0, 0], lines_2d[:, 0, 1])
        right_slope = geometry_utils.torch_line_to_orientation(
            lines_2d[:, 1, 0], lines_2d[:, 1, 1])
        visible_cond = left_slope * right_slope > 0

        near_side = self.reorder_lines(near_side)

        return near_side, visible_cond

    def encode_batch_bbox(self, gt_boxes_3d, proposals, assigned_gt_labels,
                          p2):
        """
        encoding dims may be better,here just encode dims_2d
        Args:
            dims: shape(N,6), (h,w,l) and their projection in 2d
        """

        dims = gt_boxes_3d[:, :3]
        bg_mean_dims = torch.zeros_like(self.mean_dims[:, -1:, :])
        mean_dims = torch.cat([bg_mean_dims, self.mean_dims], dim=1)
        assigned_mean_dims = mean_dims[0][assigned_gt_labels].float()
        assigned_std_dims = torch.ones_like(assigned_mean_dims)
        dims_targets = (dims - assigned_mean_dims) / assigned_std_dims

        location = gt_boxes_3d[:, 3:6]
        ry = gt_boxes_3d[:, 6:]

        label_boxes_3d = torch.cat([location, dims, ry], dim=-1)

        # corners_3d = geometry_utils.torch_boxes_3d_to_corners_3d(label_boxes_3d)
        # corners_2d = geometry_utils.torch_points_3d_to_points_2d(corners_3d.view(-1, 3), p2)

        corners_3d = geometry_utils.torch_boxes_3d_to_corners_3d(
            label_boxes_3d)
        corners_2d = geometry_utils.torch_points_3d_to_points_2d(
            corners_3d.contiguous().view(-1, 3), p2).view(-1, 8, 2)
        # find two nearest lines among eight lines
        bottom_side_lines = corners_3d[:, [0, 3, 1, 2]]
        bottom_face_lines = corners_3d[:, [0, 1, 2, 3]]
        bottom_side_lines_2d = corners_2d[:, [0, 3, 1, 2]]
        bottom_face_lines_2d = corners_2d[:, [0, 1, 2, 3]]

        side_lines, side_visible = self.find_nearest_between_two_lines(
            bottom_side_lines, bottom_side_lines_2d)
        face_lines, face_visible = self.find_nearest_between_two_lines(
            bottom_face_lines, bottom_face_lines_2d)
        # import ipdb
        # ipdb.set_trace()

        ry_targets_face = encoder_utils.encode_lines(face_lines, proposals)
        ry_targets_side = encoder_utils.encode_lines(side_lines, proposals)
        ry_targets_face = torch.cat(
            [ry_targets_face.view(-1, 4),
             face_visible.float().unsqueeze(-1)],
            dim=-1)
        ry_targets_side = torch.cat(
            [ry_targets_side.view(-1, 4),
             side_visible.float().unsqueeze(-1)],
            dim=-1)

        # depth
        depth_targets = location[:, -1:]

        # encode proj_2d
        proj_2d = geometry_utils.torch_points_3d_to_points_2d(location, p2)
        proj_2d_targets = encoder_utils.encode_points(proj_2d, proposals)

        targets = torch.cat(
            [
                dims_targets, depth_targets, proj_2d_targets, ry_targets_face,
                ry_targets_side
            ],
            dim=-1)
        return targets

    def encode_batch_angle(self, dims, assigned_gt_labels):
        """
        encoding dims may be better,here just encode dims_2d
        Args:
            dims: shape(N,6), (h,w,l) and their projection in 2d
        """

        bg_mean_dims = torch.zeros_like(self.mean_dims[:, -1:, :])
        mean_dims = torch.cat([bg_mean_dims, self.mean_dims], dim=1)
        assigned_mean_dims = mean_dims[0][assigned_gt_labels].float()
        assigned_std_dims = torch.ones_like(assigned_mean_dims)
        targets = (dims[:, :3] - assigned_mean_dims) / assigned_std_dims

        targets = torch.cat([targets, dims[:, 3:]], dim=-1)
        return targets

    def encode_batch_keypoint(self, keypoint, num_intervals, rois_batch):
        x = keypoint[:, 0]
        keypoint_type = keypoint[:, 2].long()

        rois = rois_batch[0, :, 1:]

        num_bbox = rois.shape[0]
        x_start = rois[:, 0]
        w = rois[:, 2] - rois[:, 0] + 1
        x_stride = w / num_intervals
        x_offset = torch.round((x - x_start) / x_stride).long()
        keypoint_gt = torch.zeros((num_bbox, 4 * 28)).type_as(rois_batch)
        x_index = keypoint_type * 28 + x_offset
        row_ind = torch.arange(0, num_bbox).type_as(x_index)
        keypoint_gt[row_ind, x_index] = 1
        return keypoint_gt

    def decode_batch_dims(self, targets, rois_batch):
        """
        Args:
            boxes_2d: shape(N,)
            targets: shape(N,)
        """
        # dims
        h_3d_mean = 1.67
        w_3d_mean = 1.87
        l_3d_mean = 3.7

        h_3d_std = 1
        w_3d_std = 1
        l_3d_std = 1

        h_3d = targets[:, 0] * h_3d_std + h_3d_mean
        w_3d = targets[:, 1] * w_3d_std + w_3d_mean
        l_3d = targets[:, 2] * l_3d_std + l_3d_mean

        # rois w and h
        rois = rois_batch[0, :, 1:]
        w = rois[:, 2] - rois[:, 0] + 1
        h = rois[:, 3] - rois[:, 1] + 1
        x = (rois[:, 2] + rois[:, 0]) / 2
        y = (rois[:, 3] + rois[:, 1]) / 2
        centers = torch.stack([x, y], dim=-1)
        dims = torch.stack([w, h], dim=-1)

        # import ipdb
        # ipdb.set_trace()
        points = centers.unsqueeze(1) + targets[:, 3:].view(
            -1, 4, 2) * dims.unsqueeze(1)
        # point2 = centers + targets[:, 5:7] * dims

        # cls orient
        # cls_orient = targets[:, 3:5]
        # cls_orient = F.softmax(cls_orient, dim=-1)
        # cls_orient, cls_orient_argmax = torch.max(cls_orient, dim=-1)

        # reg_orient = targets[:, 5:7]

        # decode h_2d
        # h_2d = torch.exp(targets[:, 7]) * h

        # decode c_2d
        # c_2d_x = targets[:, 8] * w + x
        # c_2d_y = targets[:, 9] * h + y

        bbox = torch.stack([h_3d, w_3d, l_3d], dim=-1)
        return torch.cat([bbox, points[:, 1], points[:, 2]], dim=-1)
        # return torch.cat([bbox, targets[:, 3:]], dim=-1)

    def decode_batch_depth(self, targets):
        h_3d_mean = 1.67
        w_3d_mean = 1.87
        l_3d_mean = 3.7

        h_3d_std = 1
        w_3d_std = 1
        l_3d_std = 1

        h_3d = targets[:, 0] * h_3d_std + h_3d_mean
        w_3d = targets[:, 1] * w_3d_std + w_3d_mean
        l_3d = targets[:, 2] * l_3d_std + l_3d_mean

        cls_orient = targets[:, 3:5]
        cls_orient = F.softmax(cls_orient, dim=-1)
        cls_orient, cls_orient_argmax = torch.max(cls_orient, dim=-1)

        reg_orient = targets[:, 5:7]

        bbox = torch.stack([h_3d, w_3d, l_3d], dim=-1)
        orient = torch.stack(
            [
                cls_orient_argmax.type_as(reg_orient), reg_orient[:, 0],
                reg_orient[:, 1]
            ],
            dim=-1)

        # decode location
        # depth_ind_preds = targets[:, 7:7 + 11]
        # depth_ind_preds = F.softmax(depth_ind_preds, dim=-1)
        # _, depth_ind_preds_argmax = torch.max(depth_ind_preds, dim=-1)

        # depth_ind = depth_ind_preds_argmax.float().unsqueeze(-1)

        # return torch.cat([bbox, orient, depth_ind, targets[:, 7 + 11:]],
        # dim=-1)
        return torch.cat([bbox, orient, targets[:, 7:]], dim=-1)

    def decode_batch_bbox(self, targets, proposals, p2):

        p2 = p2.float()
        bg_mean_dims = torch.zeros_like(self.mean_dims[:, -1:, :])
        mean_dims = torch.cat([bg_mean_dims, self.mean_dims], dim=1).float()
        std_dims = torch.ones_like(mean_dims)
        dims = targets[:, :3].view(targets.shape[0], -1,
                                   3) * std_dims + mean_dims
        dims = dims.view(targets.shape[0], -1)

        # dims_targets, depth_targets, proj_2d_targets, ry_targets_face,
        # ry_targets_side

        depth = targets[:, 3:4]
        encoded_center_2d = targets[:, 4:6]
        ry_face_pred = targets[:, 6:12]
        ry_side_pred = targets[:, 12:18]

        # center_2d = encoded_center_2d * wh + rois[:, :2]
        center_2d = encoder_utils.decode_points(encoded_center_2d, proposals)
        location = geometry_utils.torch_points_2d_to_points_3d(
            center_2d, depth, p2)

        # import ipdb
        # ipdb.set_trace()

        ry_face_visible = ry_face_pred[:, 4:]
        ry_face_pred = ry_face_pred[:, :4]
        ry_face_visible = F.softmax(ry_face_visible, dim=-1)[:, 1]
        ry_face = encoder_utils.decode_ry(ry_face_pred, proposals, p2)

        ry_side_visible = ry_side_pred[:, 4:]
        ry_side_pred = ry_side_pred[:, :4]
        # 1 refers to visible
        ry_side_visible = F.softmax(ry_side_visible, dim=-1)[:, 1]
        ry_side = encoder_utils.decode_ry(ry_side_pred, proposals, p2)

        # import ipdb
        # ipdb.set_trace()
        ry = ry_side.clone()
        ry[ry_face_visible > 0.5] = ry_face[ry_face_visible > 0.5] + 0.5 * math.pi

        # overwrite ry_face in some case
        ry[ry_side_visible > 0.5] = ry_side[ry_side_visible > 0.5]
        # ry = ry_side

        return torch.cat([dims, location, ry], dim=-1)

    def decode_batch_angle(self, targets, bin_centers=None):
        """
        Args:
            targets: shape(N, 3)
        """
        bg_mean_dims = torch.zeros_like(self.mean_dims[:, -1:, :])
        mean_dims = torch.cat([bg_mean_dims, self.mean_dims], dim=1).float()
        # assigned_mean_dims = mean_dims[0][pred_labels].float()
        std_dims = torch.ones_like(mean_dims)
        #  targets = (dims[:, :3] - assigned_mean_dims) / assigned_std_dims
        bbox = targets[:, :-2].view(targets.shape[0], -1,
                                    3) * std_dims + mean_dims
        bbox = bbox.view(targets.shape[0], -1)

        # ry
        # sin = targets[:, -2]
        # cos = targets[:, -1]
        theta = get_angle(targets[:, -1], targets[:, -2])
        if bin_centers is not None:
            theta = bin_centers + theta
            # theta = bin_centers
            # theta = -torch.acos(targets[:, 3]) - bin_centers

        # cond_pos = (cos < 0) & (sin > 0)
        # cond_neg = (cos < 0) & (sin < 0)
        # theta[cond_pos] = math.pi - theta[cond_pos]
        # theta[cond_neg] = -math.pi - theta[cond_neg]

        # ry = torch.atan(sin / cos)
        # cond = cos < 0
        # cond_pos = sin > 0
        # cond_neg = sin < 0
        # ry[cond & cond_pos] = ry[cond & cond_pos] + math.pi
        # ry[cond & cond_neg] = ry[cond & cond_neg] - math.pi

        return torch.cat([bbox, theta.unsqueeze(-1)], dim=-1)
        # bbox = torch.stack([h_3d, w_3d, l_3d, theta], dim=-1)
        # return bbox
