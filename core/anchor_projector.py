# -*- coding: utf-8 -*-

import torch


class AnchorProjector(object):
    @classmethod
    def proj_point_3to2img(cls, pts_3d, p2):
        ones = torch.ones_like(pts_3d[:, -1:])
        pts_3d_homo = torch.cat([pts_3d, ones], dim=-1).transpose(1, 0)
        pts_2d_homo = p2.matmul(pts_3d_homo).transpose(2, 1)

        pts_2d_homo = pts_2d_homo / pts_2d_homo[:, :, -1:]
        return pts_2d_homo[:, :, :-1]

    @classmethod
    def boxes2corners(cls, anchors):
        locations = anchors[:, :3]
        dims = anchors[:, 3:6]
        ry = anchors[:, 6]
        h = dims[:, 1]
        w = dims[:, 2]
        l = dims[:, 0]

        # 3d bounding box corners
        zeros = torch.zeros_like(l)
        x_corners = torch.stack(
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            dim=-1)
        y_corners = torch.stack(
            [zeros, zeros, zeros, zeros, -h, -h, -h, -h], dim=-1)
        z_corners = torch.stack(
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            dim=-1)

        # rotation matrix
        c = torch.cos(ry)
        s = torch.sin(ry)
        ones = torch.ones_like(c)
        zeros = torch.zeros_like(c)
        R = torch.stack(
            [c, zeros, s, zeros, ones, zeros, -s, zeros, c], dim=-1).view(-1,
                                                                          3, 3)

        corners_3d = torch.stack(
            [x_corners, y_corners, z_corners], dim=-1).type_as(locations)
        # shape(N,3,8)
        corners_3d = torch.bmm(R, corners_3d.permute(
            0, 2, 1)) + locations.unsqueeze(-1)
        corners_3d = corners_3d.permute(0, 2, 1).contiguous()
        return corners_3d

    @classmethod
    def project_to_image_space(cls, anchors, p2):
        corners_3d = cls.boxes2corners(anchors)

        corners_2d = cls.proj_point_3to2img(corners_3d.view(-1, 3), p2).view(
            -1, 8, 2)

        # find the min bbox
        # corners_2d: shape(N, 8, 2)
        xmin, _ = torch.min(corners_2d[:, :, 0], dim=-1)
        ymin, _ = torch.min(corners_2d[:, :, 1], dim=-1)
        xmax, _ = torch.max(corners_2d[:, :, 0], dim=-1)
        ymax, _ = torch.max(corners_2d[:, :, 1], dim=-1)
        boxes_2d = torch.stack([xmin, ymin, xmax, ymax], dim=-1)

        return boxes_2d

    @classmethod
    def project_to_bev(cls, anchors, bev_extents, ret_norm=False):
        # shape(N,8,3)
        corners_3d = cls.boxes2corners(anchors)

        # proj to bev
        x = corners_3d[:, :, 0]
        z = corners_3d[:, :, 2]
        xmin, _ = torch.min(x, dim=-1)
        xmax, _ = torch.max(x, dim=-1)
        zmin, _ = torch.min(z, dim=-1)
        zmax, _ = torch.max(z, dim=-1)

        bev_x_extents_min = bev_extents[0][0]
        bev_z_extents_min = bev_extents[1][0]
        bev_x_extents_max = bev_extents[0][1]
        bev_z_extents_max = bev_extents[1][1]
        bev_x_extents_range = bev_x_extents_max - bev_x_extents_min
        bev_z_extents_range = bev_z_extents_max - bev_z_extents_min

        z1 = bev_z_extents_max - zmax
        z2 = bev_z_extents_max - zmin
        zmin = z1
        zmax = z2

        bev_box_corners = torch.stack([xmin, zmin, xmax, zmax], dim=-1)
        # change coords origin
        bev_extents_min_tiled = [
            bev_x_extents_min, bev_z_extents_min, bev_x_extents_min,
            bev_z_extents_min
        ]
        bev_box_corners = bev_box_corners - torch.tensor(
            bev_extents_min_tiled).type_as(bev_box_corners)

        if ret_norm:
            # normalized  bev bbox
            extents_tiled = [
                bev_x_extents_range, bev_z_extents_range, bev_x_extents_range,
                bev_z_extents_range
            ]
            bev_box_corners_norm = bev_box_corners / torch.tensor(
                extents_tiled).type_as(bev_box_corners)

            return bev_box_corners, bev_box_corners_norm
        else:
            return bev_box_corners
