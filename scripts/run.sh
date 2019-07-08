#!/usr/bin/env bash

# python utils/generate_config.py

# CUDA_VISIBLE_DEVICES=1 python train.py --cuda \
    # --net faster_rcnn \
    # --out_path /data/object/liangxiong/test \
    # --config configs/test_config.json \
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
# CUDA_VISIBLE_DEVICES=1 python train.py --cuda \
    # --net fpn \
    # --out_path /data/object/liangxiong/fpn_kitti_pretrained \
    # --config configs/fpn_kitti_config.json \
    # --model /data/object/liangxiong/fpn_bdd_pretrained/fpn/bdd/detector_300000.pth

# FPN_COCO
# CUDA_VISIBLE_DEVICES=1 python train.py --cuda \
    # --net fpn \
    # --out_path /data/object/liangxiong/fpn_coco_pretrained \
    # --config configs/fpn_coco_config.json

# FPN_BDD
# CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda \
    # --net fpn \
    # --out_path /data/object/liangxiong/test \
    # --config configs/test_config.json \
    # --mGPUs
    # --resume True \
    # --checkpoint 84000
    

# MONO_3D
# CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda \
    # --net mono_3d \
    # --out_path /data/object/liangxiong/mono_3d \
    # --config configs/mono_3d_config.json

# FPN_MONO_3D (KITTI)
# CUDA_VISIBLE_DEVICES=0 python train.py --cuda \
    # --net fpn_mono_3d \
    # --out_path /data/object/liangxiong/test \
    # --config configs/test_config.json \
    # --model /data/object/liangxiong/fpn_mono_3d/fpn_mono_3d/kitti/detector_372000.pth
    # --model /data/object/liangxiong/fpn_bdd_pretrained/fpn/bdd/detector_300000.pth
    # --mGPUs
    # --mGPUs
# --resume True \
# --checkpoint 600
    # --mGPUs
# FPN_MONO_3D (NUSCENES)
# CUDA_VISIBLE_DEVICES=1 python train.py --cuda \
    # --net fpn_corners_2d \
    # --out_path /data/object/liangxiong/test \
    # --config configs/test_config.json \
    # --mGPUs
    # --model /data/object/liangxiong/fpn_mono_3d/fpn_mono_3d/kitti/detector_600000.pth \

# FAIL(not work)
# FPN_MULTIBIN
# CUDA_VISIBLE_DEVICES=1 python train.py --cuda \
    # --net fpn_multibin_mono_3d \
    # --out_path /data/object/liangxiong/test \
    # --config configs/test_config.json \
    # --model /data/object/liangxiong/fpn_bdd_pretrained/fpn/bdd/detector_300000.pth

# PRNET_BDD
# CUDA_VISIBLE_DEVICES=1 python train.py --cuda \
    # --net prnet \
    # --out_path /data/object/liangxiong/prnet_bdd_ohem \
    # --config configs/prnet_bdd_config.json \
    # --mGPUs

# CUDA_VISIBLE_DEVICES=1 python train.py --cuda \
    # --net prnet_mono_3d \
    # --out_path /data/object/liangxiong/test \
    # --config configs/test_config.json \
    # --model /data/object/liangxiong/test/prnet/kitti/detector_172000.pth


# PRNET_MONO
# CUDA_VISIBLE_DEVICES=0 python train.py --cuda \
    # --net prnet \
    # --out_path /data/object/liangxiong/test \
    # --config configs/test_config.json
    # --mGPUs
    # --model /data/object/liangxiong/prnet_bdd/prnet/bdd/detector_300000.pth
    # --resume True
    # --checkpoint 24000 \


# CUDA_VISIBLE_DEVICES=0 python train.py --cuda \
    # --net fpn_corners_stable \
    # --out_path /data/object/liangxiong/test \
    # --config configs/test_config.json \
    # --model /data/object/liangxiong/test/fpn_corners_3d/mono_3d_kitti/detector_600000.pth

# maskrcnn keypoint
# CUDA_VISIBLE_DEVICES=0 python train.py --cuda \
    # --net maskrcnn \
    # --out_path /data/object/liangxiong/test \
    # --config configs/test_config.json \
    # --model /data/object/liangxiong/test/fpn_corners_3d/mono_3d_kitti/detector_600000.pth

# CUDA_VISIBLE_DEVICES=1 python train.py --cuda \
    # --net mobileye \
    # --out_path /data/object/liangxiong/test \
    # --config configs/test_config.json \
    # --model /data/object/liangxiong/test/fpn_corners_3d/mono_3d_kitti/detector_600000.pth

# corners stable
# CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda \
    # --net fpn_corners_3d \
    # --out_path /data/object/liangxiong/test \
    # --config configs/test_config.json \
    # --model /data/object/liangxiong/test/fpn_corners_3d/mono_3d_kitti/detector_600000.pth \
    # --mGPUs

# GRNET
# CUDA_VISIBLE_DEVICES=0 python train.py --cuda \
    # --net fpn_grnet \
    # --out_path /data/object/liangxiong/test \
    # --config configs/test_config.json \
    # --model /data/object/liangxiong/test/fpn_corners_3d/mono_3d_kitti/detector_600000.pth


# CUDA_VISIBLE_DEVICES=0 python train.py --cuda \
    # --net fpn_corners_stable \
    # --out_path /data/object/liangxiong/test \
    # --config configs/test_config.json \
    # --model /data/object/liangxiong/test/fpn_corners_3d/mono_3d_kitti/detector_600000.pth

# CUDA_VISIBLE_DEVICES=0 python train.py --cuda \
    # --net fpn_mono_3d_better \
    # --out_path /data/object/liangxiong/test \
    # --config configs/test_config.json \
    # --model /data/object/liangxiong/test/fpn_corners_3d/mono_3d_kitti/detector_600000.pth
