# -*- coding: utf-8 -*-
"""
check presudo point cloud with lidar point cloud
"""

import numpy as np
import os

lidar_path = '/data/object/training/velodyne'
pesudo_lidar_path = '/data/liangxiong/KITTI/training/velodyne/'


def read_point_clouds(file_dir, sample_name):
    file_path = os.path.join(file_dir, '{}.bin'.format(sample_name))
    pc = np.fromfile(file_path, dtype=np.float32)
    return pc


def print_range(pc):
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]
    print('x range: {} - {}'.format(x.min(), x.max()))
    print('y range: {} - {}'.format(y.min(), y.max()))
    print('z range: {} - {}'.format(z.min(), z.max()))


def main():
    # import ipdb
    # ipdb.set_trace()
    sample_name = '000008'
    lidar_pc = read_point_clouds(lidar_path, sample_name).reshape(-1, 4)
    pesudo_lidar_pc = read_point_clouds(pesudo_lidar_path,
                                        sample_name).reshape(-1, 4)
    print_range(lidar_pc)
    print_range(pesudo_lidar_pc)

    # collect both of them
    np.concatenate(
        [lidar_pc, pesudo_lidar_pc],
        axis=0).astype(np.float32).tofile('000001.bin')


if __name__ == '__main__':
    main()
