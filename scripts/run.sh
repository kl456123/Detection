#!/usr/bin/env bash

# CUDA_VISIBLE_DEVICES=1 python train.py --cuda \
    # --net faster_rcnn \
    # --out_path /data/object/liangxiong/kitti_pretrained \
    # --config configs/kitti_config.json
    # --mGPUs
# --model /data/liangxiong/detection/data/pretrained_model/resnet50-19c8e357.pth


# CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda \
    # --net faster_rcnn \
    # --out_path /data/object/liangxiong/coco_pretrained \
    # --config configs/coco_config.json \
    # --mGPUs \
    # --resume True \
    # --checkpoint 68000

# CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda \
    # --net faster_rcnn \
    # --out_path /data/object/liangxiong/bdd_pretrained \
    # --config configs/bdd_config.json

# FPN_KITTI
# CUDA_VISIBLE_DEVICES=0 python train.py --cuda \
    # --net fpn \
    # --out_path /data/object/liangxiong/fpn_kitti_pretrained \
    # --config configs/fpn_kitti_config.json

# FPN_COCO
CUDA_VISIBLE_DEVICES=1 python train.py --cuda \
    --net fpn \
    --out_path /data/object/liangxiong/fpn_coco_pretrained \
    --config configs/fpn_coco_config.json

# FPN_BDD
# CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda \
    # --net fpn \
    # --out_path /data/object/liangxiong/fpn_bdd_pretrained \
    # --config configs/fpn_bdd_config.json \
    # --mGPUs


# CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda \
    # --net mono_3d \
    # --out_path /data/object/liangxiong/mono_3d \
    # --config configs/mono_3d_config.json
# --resume True \
# --checkpoint 600
    # --mGPUs
