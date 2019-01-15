#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: DuanZhixiang(zhixiangduan@deepmotion.ai)
# kitti utils

import numpy as np


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def get_lidar_in_image_fov(pc_velo,
                           calib,
                           xmin,
                           ymin,
                           xmax,
                           ymax,
                           return_more=False,
                           clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & \
               (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)

    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def project_to_image(pts_3d, P):
    ''' Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def compute_box_3d(obj, P):
    ''' Takes an object and a projection matrix (P) and projects the 3d
        bounding box into the image plane.
        Returns:
            corners_2d: (8,2) array in left image coord.
            corners_3d: (8,3) array in in rect camera coord.
    '''
    # compute rotational matrix around yaw axis
    R = roty(obj.ry)

    # 3d bounding box dimensions
    l = obj.l
    w = obj.w
    h = obj.h

    # 3d bounding box corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    #print corners_3d.shape
    corners_3d[0, :] = corners_3d[0, :] + obj.t[0]
    corners_3d[1, :] = corners_3d[1, :] + obj.t[1]
    corners_3d[2, :] = corners_3d[2, :] + obj.t[2]
    #print 'cornsers_3d: ', corners_3d
    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    #print 'corners_2d: ', corners_2d
    return corners_2d, np.transpose(corners_3d)


class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''

    def __init__(self, calib_filepath):
        calibs = self.read_calib_file(calib_filepath)

        # Camera2 to Imagary camera2
        self.C2IC = [[1, 0, 0, 0], [0, np.sqrt(2) / 2, -np.sqrt(2) / 2, 40],
                     [0, np.sqrt(2) / 2, np.sqrt(2) / 2, 0]]
        self.C2IC = np.array(self.C2IC).reshape(3, 4)

        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0, [3, 3])

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # ===========================
    # ------- 3d to 3d ----------
    # ===========================
    def project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))

    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))

    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(
            np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        '''
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = (
            (uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = (
            (uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)


class Object3d(object):
    ''' 3d object label '''

    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(data[
            2])  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12],
                  data[13])  # location (x,y,z) in camera coord.
        self.ry = data[
            14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

        self.score = data[15] if data.__len__() == 16 else 0.

        self.box3d = np.array(
            [self.ry, self.h, self.w, self.l, data[11], data[12], data[13]])

    def print_object(self):
        print(('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
              (self.type, self.truncation, self.occlusion, self.alpha)))
        print(('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
              (self.xmin, self.ymin, self.xmax, self.ymax)))
        print(('3d bbox h,w,l: %f, %f, %f' % \
              (self.h, self.w, self.l)))
        print(('3d bbox location, ry: (%f, %f, %f), %f' % \
              (self.t[0],self.t[1],self.t[2],self.ry)))


def compute_local_angle(center_2d, p2, ry):
    """
    Args:
        center_2d: shape(N, 2)
        p2: shape(3,4)
    """
    #  import ipdb
    #  ipdb.set_trace()
    M = p2[:, :3]
    center_2d_homo = np.concatenate(
        [center_2d, np.ones_like(center_2d[-1:])], axis=-1)
    direction_vector = np.dot(np.linalg.inv(M), center_2d_homo.T).T
    x_vector = np.array([1, 0, 0])
    cos = np.dot(direction_vector, x_vector.T) / np.linalg.norm(
        direction_vector, axis=-1)
    ray_angle = np.arccos(cos)
    local_angle = ry + ray_angle
    if local_angle > np.pi:
        local_angle = local_angle - 2 * np.pi
    return local_angle


def compute_global_angle(center_2d, p2, local_angle):
    """
    Note that just batch is supported
    Args:
        center_2d: shape(N, 2)
        p2: shape(3,4)
    """
    M = p2[:, :3]
    center_2d_homo = np.concatenate(
        [center_2d, np.ones_like(center_2d[:, -1:])], axis=-1)
    direction_vector = np.dot(np.linalg.inv(M), center_2d_homo.T).T
    x_vector = np.array([1, 0, 0])
    cos = np.dot(direction_vector, x_vector) / np.linalg.norm(
        direction_vector, axis=-1)
    ray_angle = np.arccos(cos)
    ry = local_angle - ray_angle
    # if ry < -np.pi:
    # ry += np.pi
    cond = ry < -np.pi
    ry[cond] = ry[cond] + 2 * np.pi
    return ry


def compute_2d_proj(ry, corners, trans_3d, p):
    import ipdb
    ipdb.set_trace()
    r = np.stack(
        [np.cos(ry), 0, np.sin(ry), 0, 1, 0, -np.sin(ry), 0, np.cos(ry)],
        axis=-1).reshape(3, 3)
    corners_3d = np.dot(r, corners.T)
    trans_3d = np.repeat(np.expand_dims(trans_3d.T, axis=1), 8, axis=1)
    corners_3d = corners_3d[..., np.newaxis] + trans_3d
    # corners_3d = corners_3d.reshape(3, -1)
    corners_3d_homo = np.vstack((corners_3d, np.ones(
        (1, corners_3d.shape[1]))))

    corners_2d = np.dot(p, corners_3d_homo)
    corners_2d_xy = corners_2d[:2] / corners_2d[2]

    corners_2d_xy = corners_2d_xy.reshape(2, 8)
    xmin = corners_2d_xy[0, :, :].min(axis=0)
    ymin = corners_2d_xy[1, :, :].min(axis=0)
    xmax = corners_2d_xy[0, :, :].max(axis=0)
    ymax = corners_2d_xy[1, :, :].max(axis=0)

    boxes_2d_proj = np.stack([xmin, ymin, xmax, ymax], axis=-1)
    return boxes_2d_proj


def truncate_box(dims_2d, line):
    """
    Args:
        dims_2d:
        line:
    Return: cls_orient:
            reg_orient:
    """
    direction = (line[0] - line[1])
    if direction[0] * direction[1] == 0:
        cls_orient = -1
    else:
        cls_orient = direction[1] / direction[0] > 0
        cls_orient = cls_orient.astype(np.int32)
    # cls_orient = direction[0] > 0
    reg_orient = np.abs(direction)

    # normalize
    w, h = dims_2d

    reg_orient[0] /= w
    reg_orient[1] /= h
    # reg_orient = np.log(reg_orient)
    return cls_orient, reg_orient


def get_h_2d(C_3d, dim, P2, box_2d):
    # x,y,z
    # C_3d = np.asarray([-16.53, 2.39, 58.49])
    # h,w,l
    # dim = np.asarray([1.67, 1.87, 3.69])

    bottom_3d = C_3d + np.asarray([0, 0.5 * dim[0], 0])
    top_3d = C_3d - np.asarray([0, 0.5 * dim[0], 0])

    bottom_3d_homo = np.append(bottom_3d, 1)
    top_3d_homo = np.append(top_3d, 1)

    bottom_2d_homo = np.dot(P2, bottom_3d_homo)
    top_2d_homo = np.dot(P2, top_3d_homo)

    lambda_bottom = bottom_2d_homo[-1]
    bottom_2d_homo = bottom_2d_homo / lambda_bottom
    bottom_2d = bottom_2d_homo[:-1]

    lambda_top = top_2d_homo[-1]
    top_2d_homo = top_2d_homo / lambda_top
    top_2d = top_2d_homo[:-1]

    delta_2d = top_2d - bottom_2d

    h = box_2d[3] - box_2d[1] + 1
    return np.abs(delta_2d[-1]) / h


def get_center_2d(C_3d, P2, box_2d):
    C_3d_homo = np.append(C_3d, 1)
    C_2d_homo = np.dot(P2, C_3d_homo)
    C_2d_homo = C_2d_homo / C_2d_homo[-1]
    C_2d = C_2d_homo[:-1]

    # normalize it by using gt box
    h = box_2d[3] - box_2d[1] + 1
    w = box_2d[2] - box_2d[0] + 1
    # x = (box_2d[3] + box_2d[1]) / 2
    # y = (box_2d[2] + box_2d[0]) / 2
    x = box_2d[0]
    y = box_2d[1]
    C_2d_normalized = ((C_2d[0] - x) / w, (C_2d[1] - y) / h)

    return C_2d_normalized
