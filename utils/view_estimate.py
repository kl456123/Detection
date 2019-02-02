# -*- coding: utf-8 -*-

import numpy as np

P2 = np.asarray([[
    7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01, 0.000000e+00,
    7.215377e+02, 1.728540e+02, 2.163791e-01, 0.000000e+00, 0.000000e+00,
    1.000000e+00, 2.745884e-03
]]).reshape((3, 4))

K = P2[:, :3]
KT = P2[:, 3]
K_inv = np.linalg.inv(K)
T = np.dot(K_inv, KT)

point_2ds = [[0, 0, 1], [1280, 384, 1], [0, 384, 1], [1280, 0, 1]]


def point2angle(point_2d, D):
    point_3d = np.dot(K_inv, point_2d) * D
    return point_3d


point_3ds = []
for point_2d in point_2ds:
    D = 80
    point_3d = point2angle(point_2d, D)
    point_3ds.append(point_3d)
    # print(point_3ds)


deltas = point_3ds[0] - point_3ds[3]
print(deltas)
