#!/bin/bash

SAMPLE_IDX=000047
CHECKEPOCH=55
NMS=0.7
THRESH=0.5

# NET=faster_rcnn
NET_NAME=double_iou_second
NET_DIR=double_iou_second

python test_net.py \
    --cuda \
    --net ${NET_NAME} \
    --checkpoint 3257 \
    --nms ${NMS} \
    --thresh ${THRESH} \
    --checkepoch ${CHECKEPOCH} \
    --load_dir /data/object/liangxiong/${NET_DIR} \
    --img_path /data/object/training/image_2/${SAMPLE_IDX}.png \
    --feat_vis True

python utils/visualize.py \
    --kitti results/data/${SAMPLE_IDX}.txt \
    --img /data/object/training/image_2/${SAMPLE_IDX}.png

