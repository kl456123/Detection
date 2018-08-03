# -*- coding: utf-8 -*-
"""
Note that this script is just writen for job clients
"""

import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in_path', default=None, type=str, help='Input directory.')
    parser.add_argument(
        '--out_path', default=None, type=str, help='Output directory.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    in_path = args.in_path
    out_path = args.out_path

    script = "trainval_net.py"
    net = "resnet50"
    config = "configs/kitti_config.json"
    command = "python {} --cuda --net {} --config {} --in_path {} --out_path"\
        .format(script, net, config, in_path, out_path)
    sys.system(command)
