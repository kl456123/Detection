#!/bin/bash

SAMPLE_IDX=000000
CHECKEPOCH=30
NMS=0.5
THRESH=0.5

# python test_net.py \
    # --cuda \
    # --net double_iou_second \
    # --checkpoint 3257 \
    # --nms ${NMS} \
    # --thresh ${THRESH} \
    # --checkepoch ${CHECKEPOCH} \
    # --load_dir /data/object/liangxiong/double_iou_second \
    # --img_path /data/object/training/image_2/${SAMPLE_IDX}.png \
    # --feat_vis True

# python utils/visualize.py \
    # --kitti results/data/${SAMPLE_IDX}.txt \
    # --img /data/object/training/image_2/${SAMPLE_IDX}.png
DIR='/home/pengwu/mono3d/kitti/0006'
for file in ${DIR}/*
do
    TMP=${file##*/}
    SAMPLE_IDX=${TMP:0:6}
    echo ${SAMPLE_IDX}
    # python test_net.py \
        # --cuda \
        # --net mono_3d \
        # --checkpoint 3257 \
        # --nms ${NMS} \
        # --thresh ${THRESH} \
        # --checkepoch ${CHECKEPOCH} \
        # --load_dir /data/object/liangxiong/mono_3d_angle_reg_3d \
        # --img_path /home/pengwu/mono3d/kitti/0006/${SAMPLE_IDX}.png \
        # --feat_vis False
    python utils/visualize.py \
        --kitti results/data/${SAMPLE_IDX}.txt \
        --img /home/pengwu/mono3d/kitti/0006/${SAMPLE_IDX}.png
done



