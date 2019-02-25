#!/bin/bash


rm results/fv/*
rm results/data/*

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
    # --checkepoch 26 \
    # --net faster_rcnn \
    # --load_dir /data/object/liangxiong/faster_rcnn \
    # --config /data/object/liangxiong/faster_rcnn/faster_rcnn/kitti/kitti_config.json

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 32 \
    # --net new_faster_rcnn \
    # --load_dir /data/object/liangxiong/exp_iouweights_hem

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 43 \
    # --net distance_faster_rcnn \
    # --load_dir /data/object/liangxiong/distance_center

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 53 \
    # --net faster_rcnn \
    # --load_dir /data/object/liangxiong/faster_rcnn_detection

# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 1 \
    # --net refine_faster_rcnn \
    # --load_dir /data/object/liangxiong/refine

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 25 \
    # --net new_faster_rcnn \
    # --load_dir /data/object/liangxiong/exp_iouweights_hem_great

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 2 \
    # --net rfcn \
    # --load_dir /data/object/liangxiong/rfcn

# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --net fpn \
    # --checkpoint 3257 \
    # --checkepoch 38 \
    # --load_dir /data/object/liangxiong/fpn

# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 88 \
    # --net semantic \
    # --load_dir /data/object/liangxiong/semantic \
    # --thresh 0.1
# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 40 \
    # --net semantic \
    # --load_dir /data/object/liangxiong/semantic_05

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 8 \
    # --net faster_rcnn \
    # --load_dir /data/object/liangxiong/use_iou

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 25 \
    # --net semantic \
    # --load_dir /data/object/liangxiong/part05

# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 24 \
    # --net semantic \
    # --load_dir /data/object/liangxiong/semantic_weights

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 38 \
    # --net three_iou \
    # --load_dir /data/object/liangxiong/three_iou_slow_ohem

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
# --net three_iou_org_ohem \
# --load_dir /data/object/liangxiong/three_iou_best_attention \
# --checkpoint 3257 \
# --checkepoch 24

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 50 \
    # --net double_iou \
    # --load_dir /data/object/liangxiong/double_iou \
    # --nms 0.7 \
    # --thresh 0.2

# no encoded
# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --net mono_3d \
    # --load_dir /data/object/liangxiong/mono_3d_train \
    # --checkpoint 3257 \
    # --checkepoch 100

# encoded
# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --net mono_3d_simpler \
    # --load_dir /data/object/liangxiong/mono_3d_angle_reg_3d_both \
    # --checkpoint 3257 \
    # --checkepoch 25

CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    --net oft \
    --load_dir /data/object/liangxiong/tmp/oft \
    --checkpoint 6683 \
    --checkepoch 18 \
    --img_dir /data/2011_09_26/2011_09_26_drive_0011_sync/image_02/data/ \
    --calib_file ./000000.txt

# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --net oft \
    # --load_dir /data/object/liangxiong/tmp/oft \
    # --checkpoint 6683 \
    # --checkepoch 11
# --calib_file ./000001.txt \
# --img_path /home/pengwu/mono3d/seq/frames/1535193200792697000.jpg \
# --feat_vis True
    # --img_dir /home/pengwu/mono3d/kitti/0006 \

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --net ssd \
    # --load_dir /data/object/liangxiong/ssd \
    # --checkpoint 3257 \
    # --checkepoch 57

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --net multibin_simpler \
    # --load_dir /data/object/liangxiong/mono_3d_angle_reg_2d \
    # --checkpoint 3257 \
    # --checkepoch 12

# 3d proj 2d detection
# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --net mono_3d \
    # --load_dir /data/object/liangxiong/mono_3d_angle_reg_3d \
    # --checkpoint 3257 \
    # --checkepoch 40
# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 42 \
    # --net three_iou_org_ohem_second \
    # --load_dir /data/object/liangxiong/three_iou_best_second


# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 2 \
    # --net single_iou \
    # --load_dir /data/object/liangxiong/single_iou

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 60 \
    # --net iou_faster_rcnn \
    # --load_dir /data/object/liangxiong/iou

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --net loss \
    # --load_dir /data/object/liangxiong/loss \
    # --checkpoint 3257 \
    # --checkepoch 7

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --net LED \
    # --load_dir /data/object/liangxiong/LED_clip \
    # --checkpoint 3257 \
    # --checkepoch 10

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --net three_iou_org_ohem \
    # --load_dir /data/object/liangxiong/delta \
    # --checkpoint 3257 \
    # --checkepoch 18

# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 82 \
    # --net overlaps \
    # --load_dir /data/object/liangxiong/overlaps
