# -*- coding: utf-8 -*-

from wavedata.tools.obj_detection import obj_utils
from wavedata.tools.core import calib_utils
import numpy as np
from utils import geometry_utils
import os

import matplotlib.pyplot as plt

import cv2
import pickle


def cam_pts_to_rect_depth(cam_pts: np.ndarray, K: np.ndarray, h: int, w: int):
    assert cam_pts.shape[0] == 3  # (3, N)

    y, x, depth, sel = cam_pts_to_rect_depth_mapping(cam_pts, K, h, w, False)

    mat = np.zeros((h, w), dtype=np.float32)
    mat[y[sel], x[sel]] = depth[sel]
    return mat


def cam_pts_to_rect_depth_mapping(cam_pts: np.ndarray,
                                  K: np.ndarray,
                                  h: int,
                                  w: int,
                                  debug=False):

    assert cam_pts.shape[0] == 3  # (3, N)

    uv_depth = K @ cam_pts
    depth = uv_depth[2, :]
    uv_depth[:2, :] /= depth

    x = np.rint(uv_depth[0, :]).astype(np.int)
    y = np.rint(uv_depth[1, :]).astype(np.int)

    sel = ((x >= 0) & (x < w) & (y >= 0) & (y < h) & (depth > 0))

    return y, x, depth, sel


def draw_points_cloud(pc):
    pcd = pc2pcd(pc)
    draw_geometries([pcd])


def pc2pcd(pc):
    """
    Args:
        pc: shape(3,N)
    """

    assert pc.shape[0] == 3
    pcd = PointCloud()
    pcd.points = Vector3dVector(pc.T)
    return pcd


def reverse_project_depth_w_color(depth: np.ndarray,
                                  im: np.ndarray = None,
                                  K_cam: np.ndarray = np.eye(3)):
    assert len(depth.shape) == 2  # (H, W)
    y, x = np.where(depth > 0.0)
    uv1 = np.vstack((x, y, np.ones_like(x, dtype=np.float32)))  # (3, N)
    uv_depth = uv1 * depth[y, x]
    pts_cam = np.linalg.inv(K_cam) @ uv_depth

    if im is not None:
        color = im[y, x]
    else:
        color = None

    return pts_cam, color


def pc2color_pcd(pc, color):
    from open3d import PointCloud, Vector3dVector
    color = np.array(color).reshape(-1, 3)
    assert pc.shape[0] == 3
    if color.shape[0] == 1:
        color = np.repeat(color, pc.shape[1], axis=0)

    color = color.copy().astype(np.float32)
    color /= 255.0

    pcd = PointCloud()
    pcd.points = Vector3dVector(pc.T)
    pcd.colors = Vector3dVector(color[:, ::-1])
    return pcd


def visualize_pointcloud(image_path,
                         boxes_2d,
                         instance_depth_map,
                         p2,
                         win_name='test'):
    # import ipdb
    # ipdb.set_trace()
    # h, w, _ = im.shape
    sample_name = os.path.basename(image_path)[:-4]
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    depth_map = np.zeros((h, w))

    boxes_2d = boxes_2d.astype(np.int)
    boxes_2d[boxes_2d < 0] = 0
    boxes_2d[:, 2] = np.minimum(boxes_2d[:, 2], w - 1)
    boxes_2d[:, 3] = np.minimum(boxes_2d[:, 3], h - 1)
    center_depth = instance_depth_map.mean(axis=-1)
    ascending_order = np.argsort(center_depth)
    descending_order = ascending_order[::-1]
    instance_depth_map = instance_depth_map.reshape(-1, 28, 28)
    # reorder
    instance_depth_map = instance_depth_map[descending_order]
    boxes_2d = boxes_2d[descending_order]
    for box_id, box_2d in enumerate(boxes_2d):
        x1, y1, x2, y2 = box_2d
        x1 = min(x1, w - 1)
        x2 = min(x2, w - 1)
        y1 = min(y1, h - 1)
        depth_map[y1:y2, x1:x2] = cv2.resize(
            instance_depth_map[box_id], (x2 - x1, y2 - y1),
            interpolation=cv2.INTER_CUBIC)
    # plt.imshow(depth_map)
    # plt.show()

    pkl_path = os.path.join('./results/pkls', '{}.pkl'.format(sample_name))
    K_camX = p2[:3, :3]
    res = {}
    res['image'] = image
    res['depth'] = depth_map
    res['K_camX'] = K_camX
    with open(pkl_path, 'wb') as f:
        pickle.dump(res, f)
    # velo_back, color = reverse_project_depth_w_color(depth_map, image, K_camX)
    # velo_back = velo_back[:3, :]

    # pcd_back = pc2color_pcd(velo_back, color)
    # draw_geometries([pcd_back])


def local_visualize():
    from open3d import PointCloud, Vector3dVector
    from open3d import draw_geometries
    pkls_dir = './results/pkls/'
    for pkl in sorted(os.listdir(pkls_dir)):
        pkl_path = os.path.join(pkls_dir, pkl)
        with open(pkl_path, 'rb') as f:
            res = pickle.load(f)
        depth_map = res['depth']
        # depth_map[...] = 10
        image = res['image']
        K_camX = res['K_camX']
        velo_back, color = reverse_project_depth_w_color(
            depth_map, image, K_camX)
        velo_back = velo_back[:3, :]

        pcd_back = pc2color_pcd(velo_back, color)
        draw_geometries([pcd_back])


def test():
    pass


# img_idx = 0
# calib_dir = '/data/object/training/calib'
# velo_dir = '/data/object/training/velodyne'
# img_dir = '/data/object/training/image_02'
# im_size = [384, 1280]
# # import ipdb
# # ipdb.set_trace()
# pc = obj_utils.get_lidar_point_cloud(
# img_idx, calib_dir, velo_dir, im_size=None, min_intensity=None)
# # pts, colors = obj_utils.get_lidar_point_cloud_with_color(img_idx,
# # img_dir,
# # calib_dir,
# # velo_dir,
# # im_size=None)
# print(pc.shape)
# # draw_points_cloud(pc)

# # im = load_im(cam_f)
# # velo = load_velodyne(velo_f, no_reflect=True).transpose()

# # h, w, _ = im.shape
# h, w = im_size

# # import ipdb
# # ipdb.set_trace()
# # cam_pts = T_camX_velo @ cart_to_homo(velo)
# frame_calib = calib_utils.read_calibration(calib_dir, img_idx)
# p2 = frame_calib.p2
# K_camX, T = geometry_utils.ProjectMatrixTransform.decompose_matrix(p2)
# depth = cam_pts_to_rect_depth(pc[:3, :], K=K_camX, h=h, w=w)
# plt.imshow(depth)
# plt.show()
# print(depth.shape)


def box_3d_filter(box_3d, pointcloud):
    # planes = geometry_utils.boxes_3d_to_plane(box_3d[None])[0]
    corners_3d = geometry_utils.boxes_3d_to_corners_3d(box_3d[None])[0]
    point_mask = obj_utils.is_point_inside(pointcloud, corners_3d.T)

    return pointcloud.T[point_mask].T


def two_plane_filter(planes, point_cloud):
    """
    filter inside points
    """
    front_plane = planes[0]
    rear_plane = planes[1]
    t = np.linalg.norm(front_plane[:-1]) / np.linalg.norm(rear_plane[:-1])
    if front_plane[-1] > rear_plane[-1] * t:
        rear_plane_filter = plane_filter(-rear_plane, point_cloud)
        front_plane_filter = plane_filter(front_plane, point_cloud)
        return rear_plane_filter & front_plane_filter
    else:
        rear_plane_filter = plane_filter(rear_plane, point_cloud)
        front_plane_filter = plane_filter(-front_plane, point_cloud)
        return rear_plane_filter & front_plane_filter


def plane_filter(ground_plane, point_cloud, offset_dist=0):
    ground_plane = np.array(ground_plane)

    # Calculate filter using ground plane
    ones_col = np.ones(point_cloud.shape[1])
    padded_points = np.vstack([point_cloud, ones_col])

    offset_plane = ground_plane + [0, 0, 0, -offset_dist]

    # Create plane filter
    dot_prod = np.dot(offset_plane, padded_points)
    plane_filter = dot_prod > 0

    return plane_filter

    # Combine the two filters
    # point_filter = np.logical_and(extents_filter, plane_filter)


if __name__ == '__main__':
    local_visualize()
    # test()
