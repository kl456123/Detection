#!/bin/bash

SAMPLE_IDX=000063
# SAMPLE_IDX=000128
# SAMPLE_IDX=000047
SAMPLE_IDX=000008
CHECKEPOCH=48

# dont use rpn
NMS=1
THRESH=0

# use results which stage
USE_WHICH_RESULT=none
FAKE_MATCH_THRESH=0.7

# NET=semantic
# NET_NAME=detach_double_iou
NET_NAME=post_iou
NET_DIR=post_iou
USE_GT=True

# use model directly(prior if available)
MODEL_PATH=/data/object/liangxiong/tmp/faster_rcnn_189_3257.pth
CONFIG_PATH=/data/object/liangxiong/tmp/post_iou_config.json



CUDA_VISIBLE_DEVICES=0 python test_net.py \
    --cuda \
    --net ${NET_NAME} \
    --nms ${NMS} \
    --thresh ${THRESH} \
    --load_dir /data/object/liangxiong/${NET_DIR} \
    --img_path /data/object/training/image_2/${SAMPLE_IDX}.png \
    --checkpoint 3257 \
    --checkepoch ${CHECKEPOCH} \
    --use_which_result ${USE_WHICH_RESULT} \
    --fake_match_thresh ${FAKE_MATCH_THRESH} \
    --use_gt ${USE_GT}
    # --model ${MODEL_PATH} \
    # --config ${CONFIG_PATH} \
    # --feat_vis True

# vis pred
python utils/visualize.py \
    --kitti results/data/${SAMPLE_IDX}.txt \
    --img /data/object/training/image_2/${SAMPLE_IDX}.png \
    --title pred

# vis rois
python utils/visualize.py \
    --kitti results/rois/${SAMPLE_IDX}.txt \
    --img /data/object/training/image_2/${SAMPLE_IDX}.png \
    --title rois

# vis anchors
python utils/visualize.py \
    --kitti results/anchors/${SAMPLE_IDX}.txt \
    --img /data/object/training/image_2/${SAMPLE_IDX}.png \
    --title anchors

