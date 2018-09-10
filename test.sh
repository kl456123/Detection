#!/bin/bash
# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 2 \
    # --net faster_rcnn \
    # --load_dir /data/object/liangxiong/nofocal \
    # --config /data/object/liangxiong/nofocal/faster_rcnn/kitti/kitti_config.json
    # --rois_vis

# baseline 89.2
# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 53 \
    # --net faster_rcnn \
    # --load_dir /data/object/liangxiong/faster_rcnn \
    # --config /data/object/liangxiong/faster_rcnn/faster_rcnn/kitti/kitti_config.json

CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    --checkpoint 3257 \
    --checkepoch 46 \
    --net faster_rcnn \
    --load_dir /data/object/liangxiong/scale
