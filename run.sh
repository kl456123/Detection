#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
--net faster_rcnn \
--out_path /data/object/liangxiong/scale \
--config configs/kitti_config.json
# --r True \
# --checkpoint 3257 \
# --checkepoch 1
