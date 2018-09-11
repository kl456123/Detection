#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
# --net two_rpn \
# --out_path /data/object/liangxiong/two_rpn \
# --config configs/two_rpn_config.json
# --r True \
# --checkpoint 3257 \
# --checkepoch 1


CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
--net faster_rcnn \
--out_path /data/object/liangxiong/scale_encode \
--config configs/kitti_config.json
