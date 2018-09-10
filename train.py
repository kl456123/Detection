# -*- coding: utf-8 -*-
"""
Note that this script is just writen for job clients
"""

import argparse
import time
import subprocess


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
    print(in_path)
    print(out_path)

    script = "trainval_net.py"
    net = "two_rpn"
    config = "configs/kitti_config.json"
    command = "/node01/jobs/io/env/py3torch0.4/bin/python {} --cuda --net {} --config {} --in_path {} --out_path"\
        .format(script, net, config, in_path, out_path)

    # import sys
    print("now time: ", time.time())
    # sys.system(command)
    subprocess.call(command, shell=True)
