#!/bin/bash

rm results/data/*
rm results/fv/*

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

# CUDA_VISIBLE_DEVICES=0 python test_net.py --cuda \
    # --net semantic_sinet \
    # --model ./faster_rcnn_32_3257.pth \
    # --config ./configs/refine_kitti_config.json

# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --net semantic \
    # --load_dir /data/object/liangxiong/SemanticRes101 \
    # --model /data/object/liangxiong/SemanticRes101/faster_rcnn_20_3257.pth \
    # --config /data/object/liangxiong/SemanticRes101/refine_kitti_config.json
    # --checkpoint 3257 \
    # --checkepoch 55
    # --config configs/refine_kitti_config.json

# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 20 \
    # --net semantic_sinet \
    # --load_dir /data/object/liangxiong/semantic_sdp \
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

# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --net post_cls \
    # --load_dir /data/object/liangxiong/detach_double_iou_cls_better \
    # --checkpoint 3257 \
    # --checkepoch 53

# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --net post_iou \
    # --use_gt True \
    # --model /data/object/liangxiong/tmp/faster_rcnn_189_3257.pth \
    # --config /data/object/liangxiong/tmp/post_iou_config.json

# CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    # --checkpoint 3257 \
    # --checkepoch 24 \
    # --net detach_double_iou \
    # --load_dir /data/object/liangxiong/detach_double_iou_cls
    # --nms 0.7 \
    # --thresh 0.2
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

# CUDA_VISIBLE_DEVICES=0 python test.py --cuda \
    # --checkpoint 300000 \
    # --net mono_3d \
    # --load_dir /data/object/liangxiong/mono_3d \
    # --dataset kitti

# BDD
# CUDA_VISIBLE_DEVICES=0 python test.py --cuda \
    # --checkpoint 8000 \
    # --net faster_rcnn \
    # --load_dir /data/object/liangxiong/bdd_pretrained \
    # --dataset bdd
#KITTI
# CUDA_VISIBLE_DEVICES=0 python test.py --cuda \
    # --checkpoint 32000 \
    # --net faster_rcnn \
    # --load_dir /data/object/liangxiong/kitti_pretrained \
    # --dataset kitti

#FPN_KITTI
# CUDA_VISIBLE_DEVICES=1 python test.py --cuda \
    # --checkpoint 216000 \
    # --net fpn \
    # --load_dir /data/object/liangxiong/fpn_kitti_pretrained \
    # --dataset kitti \
    # --thresh 0.5

# FPN_COCO
# CUDA_VISIBLE_DEVICES=1 python test.py --cuda \
    # --checkpoint 300000 \
    # --net fpn \
    # --load_dir /data/object/liangxiong/fpn_coco_pretrained \
    # --dataset coco

# FPN_BDD
# CUDA_VISIBLE_DEVICES=1 python test.py --cuda \
    # --checkpoint 300000 \
    # --net fpn \
    # --load_dir /data/object/liangxiong/fpn_bdd_pretrained \
    # --dataset bdd \
    # --thresh 0.5

# FPN_MONO_3D
# CUDA_VISIBLE_DEVICES=1 python test.py --cuda \
    # --checkpoint 1000 \
    # --net fpn_mono_3d \
    # --load_dir /data/object/liangxiong/fpn_mono_3d \
    # --dataset kitti \
    # --thresh 0.5
    # --img_dir /data/dm202_3w/left_img \
    # --calib_file ./000004.txt

# FPN FPN_MULTIBIN
CUDA_VISIBLE_DEVICES=1 python test.py --cuda \
    --checkpoint 1000 \
    --net fpn_multibin_mono_3d \
    --load_dir /data/object/liangxiong/fpn_multibin_mono_3d \
    --dataset kitti \
    --thresh 0.5
