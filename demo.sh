#!/bin/bash

SAMPLE_IDX=000008
CHECKEPOCH=10
NMS=0.7
THRESH=0.2

python test_net.py \
    --cuda \
    --net double_iou_second \
    --checkpoint 3257 \
    --nms ${NMS} \
    --thresh ${THRESH} \
    --checkepoch ${CHECKEPOCH} \
    --load_dir /data/object/liangxiong/double_iou_second \
    --img_path /data/object/training/image_2/${SAMPLE_IDX}.png \
    --feat_vis True

python utils/visualize.py \
    --kitti results/data/${SAMPLE_IDX}.txt \
    --img /data/object/training/image_2/${SAMPLE_IDX}.png

