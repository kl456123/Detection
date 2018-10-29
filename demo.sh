#!/bin/bash

SAMPLE_IDX=000008
CHECKEPOCH=80
NMS=0.5
THRESH=0.1

python test_net.py \
    --cuda \
    --net semantic \
    --checkpoint 3257 \
    --nms ${NMS} \
    --thresh ${THRESH} \
    --checkepoch ${CHECKEPOCH} \
    --load_dir /data/object/liangxiong/semantic/ \
    --img_path /data/object/training/image_2/${SAMPLE_IDX}.png

python utils/visualize.py \
    --kitti results/data/${SAMPLE_IDX}.txt \
    --img /data/object/training/image_2/${SAMPLE_IDX}.png

