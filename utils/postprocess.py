import numpy as np
from numpy.linalg import norm


def mono_3d_postprocess(dets_3d, p2):
    """
    Post process for generating 3d object
    Args:
    dets_3d: list of detection result for 3d reconstruction
    """
    # in single image
    # shape(N,7), (fblx,fbly,fbrx,fbry,rblx,rbly,ftly)
    rcnn_3ds = dets_3d
    p2 = p2[0]
    box_3d = []
    for i in range(rcnn_3ds.shape[0]):
        points_2d = rcnn_3ds[i, :-1].reshape((3, 2))
        y_2d = rcnn_3ds[i, -1]
        points_3d = point_2dto3d(points_2d, p2)
        # points_3d = points_3d.reshape((3, 3))

        points_3d = rect(points_3d)

        ftl = np.stack([rcnn_3ds[i, 0], y_2d], axis=-1)
        ftl_3d = get_top_side(points_3d, ftl, p2)

        # shape(N,4,3) (fbl,fbr,rbl,ftl)
        points_3d = np.concatenate([points_3d, ftl_3d.reshape(1, 3)], axis=0)

        # 3d bbox reconstruction has already done, get rlhwxyz
        l_v = points_3d[2] - points_3d[0]
        h_v = points_3d[3] - points_3d[0]
        w_v = points_3d[1] - points_3d[0]

        l = norm(l_v, axis=-1, keepdims=True)
        h = norm(h_v, axis=-1, keepdims=True)
        w = norm(w_v, axis=-1, keepdims=True)

        center = (l_v + h_v + w_v) / 2 + points_3d[0]

        # x_corners = np.array([l / 2, l / 2, -l / 2, l / 2])
        # y_corners = np.zeros(4, h.shape(0))
        # y_corners[3, :] = -h
        # # y_corners = np.array([0, 0, 0, -h])
        # z_corners = np.array([w / 2, -w / 2, w / 2, w / 2])

        # import ipdb
        # ipdb.set_trace()
        # box = np.vstack((x_corners, y_corners, z_corners))
        # box_rotated = points_3d - center
        # R = np.dot(box_rotated[:, 1:], np.linalg.inv(box[:, 1:]))
        # ry = np.arccos(R[0])
        # just for simplification
        ry = np.arccos(l_v[0] / l)
        if l_v[2] / l < 0:
            ry = -ry

        # bottom center is the origin of object coords frame
        center -= h_v / 2
        # hwlxyzry
        box_3d.append(np.concatenate([h, w, l, center, ry], axis=-1))
    return np.stack(box_3d, axis=0)


def get_top_side(points_3d, ftl, p2):
    """
    Args:
    points_3d: shape(N,3,3)
    h_2d: shape(N,)
    ftl: shape(N,2) point in 2d
    """
    # homo
    ftl_homo = np.concatenate([ftl, np.ones_like(ftl[:1])], axis=-1)
    normal = points_3d[0] - points_3d[2]
    d = -np.dot(points_3d[0], normal.T)

    p2_inv, c = decompose(p2)

    # get direction
    direction_3d = np.dot(p2_inv, ftl_homo.T)

    # c = get_camera_center(p2)
    coeff = -(np.dot(normal, c) + d) / np.dot(normal, direction_3d)
    ftl_3d = c + coeff * direction_3d

    # transform back from homo
    # ftl_3d /= ftl_3d[:, -1]
    return ftl_3d


def null(A, eps=1e-12):
    import scipy
    from scipy import linalg, matrix
    u, s, vh = scipy.linalg.svd(A)
    padding = max(0, np.shape(A)[1] - np.shape(s)[0])
    null_mask = np.concatenate(
        ((s <= eps), np.ones(
            (padding, ), dtype=bool)), axis=0)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return scipy.transpose(null_space)


def rect(points_3d):
    """
    Args:
    points_3d: shape(N,3,3)
    points_3d: shape(N,4,3)
    """
    center = (points_3d[2] + points_3d[1]) / 2
    d1 = points_3d[0] - center
    d2 = points_3d[1] - center
    deltas = (norm(d1) - norm(d2)) / 2
    fbl = center + d1 * (1 - deltas / norm(d1))
    fbr = center + d2 * (1 + deltas / norm(d2))
    rbl = center - d2 * (1 + deltas / norm(d2))
    points_3d_rect = np.zeros_like(points_3d)
    points_3d_rect[0] = fbl
    points_3d_rect[1] = fbr
    points_3d_rect[2] = rbl
    return points_3d_rect


def get_ground_plane():
    return np.asarray([0, 1, 0, -2.39])


def get_camera_center(p2):
    return null(p2).reshape((-1, 1))


def decompose(p2):
    M = p2[:, :-1]
    t = p2[:, -1]
    M_inv = np.linalg.inv(M)
    C = -np.dot(M_inv, t)
    return M_inv, C


def point_2dto3d(points_2d, p2):
    """
    Args:
    points_2d: shape(N,2), points in 2d image
    points_3d: shape(N,3), points in camera coords frame
    """
    # p2_inv = np.linalg.pinv(p2)
    p2_inv, c = decompose(p2)

    points_2d_homo = np.concatenate(
        [points_2d, np.ones_like(points_2d[:, :1])], axis=-1)

    # get direction
    direction_3d = np.dot(p2_inv, points_2d_homo.T)

    ground = get_ground_plane()

    # coeff = -np.dot(ground.T, direction_3d) / np.dot(ground.T, c)
    coeff = -(np.dot(ground[:-1].T, c) + ground[-1]) / (np.dot(ground[:-1].T,
                                                               direction_3d))
    points_3d = c + (coeff * direction_3d).T

    points_3d = points_3d

    # transform back from homo
    # points_3d /= points_3d[:, -1:]
    return points_3d
