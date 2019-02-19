# -*- coding: utf-8 -*-
"""
Convert disp/depth map to point cloud
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc
import cv2
from utils.view_estimate import draw_3d

# kitti params
baseline = 0.54
p2 = np.asarray([[
    7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01, 0.000000e+00,
    7.215377e+02, 1.728540e+02, 2.163791e-01, 0.000000e+00, 0.000000e+00,
    1.000000e+00, 2.745884e-03
]]).reshape((3, 4))
# f = p2[0, 0]

K = p2[:3, :3]
KT = p2[:, 3]
K_inv = np.linalg.inv(K)
T = np.dot(K_inv, KT)
C = -T

MAX_DEPTH = 100
original_width = 1280
original_height = 384

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351
width_to_focal[1280] = 721.5377

f = width_to_focal[original_width]


def read_img(img_name):
    img = plt.imread(img_name)
    # img = np.asarray(Image.open(img_name))
    return img


def load_npy(file_name):
    pred_disp = np.load(file_name)
    pred_disp = original_width * cv2.resize(
        pred_disp, (original_width, original_height),
        interpolation=cv2.INTER_LINEAR)
    # disp_to_img = scipy.misc.imresize(disp_pp.squeeze(),
    # [original_height, original_width])
    return pred_disp


def disp2pc(img):
    disp = np.copy(img)
    disp = disp.flatten()
    disp[disp == 0] = MAX_DEPTH
    h, w = img.shape[:2]
    u_index, v_index = np.meshgrid(range(w), range(h))
    u_index = u_index.flatten()
    v_index = v_index.flatten()
    ones = np.ones_like(u_index)
    point_2ds = np.vstack([u_index, v_index, ones])

    depth = f * baseline / disp

    pc = depth[..., np.newaxis] * np.dot(K_inv, point_2ds).T

    # translation(no rotation in kitti)
    pc = pc + C
    return pc


def disp2depth(img):
    disp = np.copy(img)
    disp[disp == 0] = MAX_DEPTH
    depth = f * baseline / disp
    return depth


def pc2bev(pc):
    pc_bev = pc[:, [0, 2]]
    voxel_size = 0.05
    width = 80
    height = 75
    bev_width = int(height / voxel_size)
    bev_height = int(width / voxel_size)
    bev = np.zeros((bev_height, bev_width))
    pc_bev[:, 0] += height / 2

    # voxelize
    pc_bev /= voxel_size
    pc_bev = pc_bev.astype(np.int32)
    area_filter = (pc_bev[:, 1] < bev_width) & (pc_bev[:, 0] < bev_height)
    zeros_filter = (pc_bev[:, 1] >= 0) & (pc_bev[:, 0] >= 0)
    pc_bev = pc_bev[area_filter & zeros_filter]
    bev[pc_bev[:, 0], pc_bev[:, 1]] = 1
    return bev


if __name__ == '__main__':
    import ipdb
    ipdb.set_trace()
    img_name = './000008_disp.npy'
    # img = read_img(img_name)
    img = load_npy(img_name)
    pc = disp2pc(img)
    depth = disp2depth(img)

    # depth
    depth_name = './depth.png'
    plt.imsave(depth_name, depth)

    # disp
    disp_name = './disp.png'
    plt.imsave(disp_name, img)

    # import ipdb
    # ipdb.set_trace()
    bev = pc2bev(pc)

    bev_name = './bev.png'
    plt.imsave(bev_name, bev)

    one = np.ones_like(pc[:, -1:])
    pc = np.concatenate([pc, one], axis=-1)

    # np.save('000008_pc.npy', pc)
    pc.astype(np.float32).tofile('000001.bin')
    # draw_3d(pc)
