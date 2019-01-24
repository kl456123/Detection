# -*- coding: utf-8 -*-
import os
import sys
from utils.box_vis import mainv2 as box_3d_vis
import time

data_dir = '/data/object/training'

result_dir = './results/data'


def read_dir(result_dir):
    pass


def main():
    data_file = 'val.txt'
    data_file_path = os.path.join(data_dir, data_file)
    with open(data_file_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    start = time.time()

    for ind, det_file in enumerate(sorted(os.listdir(result_dir))):
        sample_name = os.path.splitext(det_file)[0]

        #  import ipdb
        #  ipdb.set_trace()
        #  if not int(sample_name)==27:
        #  continue

        #  kitti_path = os.path.join(data_dir,
        #  'label_2/{}.txt'.format(sample_name))
        kitti_path = os.path.join(result_dir, '{}.txt'.format(sample_name))
        img_path = os.path.join(data_dir, 'image_2/{}.png'.format(sample_name))
        calib_path = os.path.join(data_dir, 'calib/{}.txt'.format(sample_name))
        save_path = '{}.png'.format(sample_name)
        #  command = 'python utils/box_vis.py --kitti {} --img {} --calib {} --save_path {}'.format(
        #  kitti_path, img_path, calib_path, save_path)
        box_3d_vis(img_path, kitti_path, calib_path, save_path)
        #  command = 'echo asdg'
        #  os.system(command)
        duration = time.time() - start
        sys.stdout.write(
            '\r{}/{} duration: {:.4f}'.format(ind, 3679, duration))
        sys.stdout.flush()
        #  input("Press Enter to continue...")


if __name__ == '__main__':
    main()
