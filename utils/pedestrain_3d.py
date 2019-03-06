# -*- coding: utf-8 -*-
"""
use geometry constrain to detection pedestrain in 3d space
"""

import json
import time
import sys
import numpy as np

from builder.dataloader_builders.kitti_mono_3d_dataloader_builder import Mono3DKittiDataLoaderBuilder
from utils.postprocess import mono_3d_postprocess_bbox as postprocess
from core.tester import save_dets


def read_config_json(json_file):
    with open(json_file) as f:
        config = json.load(f)
    return config


def main():
    data_config_file = './configs/pedestrain_kitti_config.json'
    data_config = read_config_json(data_config_file)
    data_loader_builder = Mono3DKittiDataLoaderBuilder(
        data_config['eval_data_config'], training=True)
    data_loader = data_loader_builder.build()
    num_samples = len(data_loader)

    for i, data in enumerate(data_loader):
        start_time = time.time()
        img_file = data['img_name']
        dets = []
        gt_boxes = data['gt_boxes'][0].cpu().numpy()
        gt_boxes_3d = data['gt_boxes_3d'][0].cpu().numpy()
        cls_dets_gt = np.concatenate(
            [gt_boxes, np.zeros_like(gt_boxes[:, -1:])], axis=-1)
        p2 = data['p2'][0].detach().cpu().numpy()

        rcnn_3d_gt, _ = postprocess(gt_boxes_3d[:, :3], cls_dets_gt, p2)
        dets.append(np.concatenate([cls_dets_gt, rcnn_3d_gt], axis=-1))
        save_dets(dets[0], img_file[0], 'kitti', 'results/data')

        duration_time = time.time() - start_time
        sys.stdout.write(
            '\r{}/{},duration: {}'.format(i + 1, num_samples, duration_time))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
