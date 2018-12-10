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
    parser.add_argument(
        '--pretrained_path',
        default='',
        type=str,
        help='Path to pretained model')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import sys
    print(sys.version)
    args = parse_args()
    import os
    print(os.path.isfile('./core/models/iou_faster_rcnn_model.py'))
    in_path = args.in_path
    out_path = args.out_path
    print(args.pretrained_path)
    print(in_path)
    # for file in os.listdir('/data1'):
    # print(file)
    print(out_path)
    #  dir_name = 'semantic_dsp_res101'
    #  out_path = os.path.join(out_path, dir_name)
    model_path = os.path.join(args.pretrained_path, 'faster_rcnn_189_3257.pth')

    script = "trainval_net.py"
    net = "post_iou"
    config = "configs/job_post_cls_config.json"
    command = "/node01/jobs/io/env/py3torch0.4/bin/python {} --cuda --net {} --config {} --in_path {} --out_path {} --model {}"\
        .format(script, net, config, in_path, out_path,model_path)

    # import sys
    print("now time: ", time.time())
    # sys.system(command)
    subprocess.call(command, shell=True)
