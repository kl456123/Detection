# -*- coding: utf-8 -*-
"""
Check if projector works
"""
import numpy as np
from utils.box_vis import compute_box_3d

kitti_label = np.asarray(
    [[4.7500, 1.4821, 3.2500, 3.4000, 1.7000, 1.5000, 1.5708]])

# P2 = np.asarray([[
# 7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01, 0.000000e+00,
# 7.215377e+02, 1.728540e+02, 2.163791e-01, 0.000000e+00, 0.000000e+00,
# 1.000000e+00, 2.745884e-03
# ]]).reshape((3, 4))
P2 = np.asarray([[721.5377, 0.0000, 609.5593, 44.8573],
                 [0.0000, 721.5377, 172.8540, 0.2164],
                 [0.0000, 0.0000, 1.0000, 0.0027]]).reshape((3, 4))

K = P2[:3, :3]
KT = P2[:, 3]
T = np.dot(np.linalg.inv(K), KT)
C = -T


def boxes2corners(boxes):
    dims = boxes[:, 3:6]
    dims = np.stack([dims[:,1],dims[:,2],dims[:,0]],axis=-1)
    pos = boxes[:, :3]
    ry = boxes[:, -1:]
    target = {'dimension': dims[0], 'location': pos[0], 'ry': ry[0]}
    corners_2d = compute_box_3d(target, P2)
    xmin = corners_2d[:, 0].min()
    xmax = corners_2d[:, 0].max()
    ymin = corners_2d[:, 1].min()
    ymax = corners_2d[:, 1].max()
    print(xmin, ymin, xmax, ymax)


boxes2corners(kitti_label)
