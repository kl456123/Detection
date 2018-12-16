#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net two_rpn \
    # --out_path /data/object/liangxiong/two_rpn \
    # --config configs/two_rpn_config.json
# --r True \
    # --checkpoint 3257 \
    # --checkepoch 1


# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net distance_faster_rcnn \
    # --out_path /data/object/liangxiong/distance \
    # --config configs/kitti_config.json

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net distance_faster_rcnn \
    # --out_path /data/object/liangxiong/distance_center \
    # --config configs/distance_center_kitti_config.json

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net rfcn \
    # --out_path /data/object/liangxiong/rfcn \
    # --config configs/rfcn_kitti_config.json
# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net double_iou_second \
    # --out_path /data/object/liangxiong/double_iou_second \
    # --config configs/double_iou_kitti_config.json

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net double_iou_second \
    # --out_path /data/object/liangxiong/double_iou_second \
    # --config configs/double_iou_kitti_config.json \
    # --checkpoint 3257 \
    # --checkepoch 13 \
    # --r True

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net fpn \
    # --out_path /data/object/liangxiong/fpn \
    # --config configs/fpn_config.json

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net double_iou_second \
    # --out_path /data/object/liangxiong/double_iou_third \
    # --config configs/double_iou_kitti_config.json \
    # --checkpoint 3257 \
    # --checkepoch 1 \
    # --r True
# --model /data/object/liangxiong/double_iou_second/double_iou_second/kitti/faster_rcnn_45_3257.pth

# --model /data/object/liangxiong/double_iou_new/double_iou/kitti/faster_rcnn_100_3257.pth
# --checkpoint 3257 \
    # --checkepoch 42 \
    # --r True
# --model /data/object/liangxiong/double_iou_new/double_iou/kitti/faster_rcnn_100_3257.pth
# --checkpoint 3257 \
    # --checkepoch 62 \
    # --r True
# --model /data/object/liangxiong/double_iou/double_iou/kitti/faster_rcnn_53_3257.pth \
    # --lr 1e-5

# no encoded
# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net mono_3d \
    # --out_path /data/object/liangxiong/mono_3d_train \
    # --config configs/refine_kitti_config.json \
    # --checkpoint 3257 \
    # --checkepoch 100

# encode
CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    --net mono_3d \
    --out_path /data/object/liangxiong/mono_3d_train_encode \
    --config configs/refine_kitti_config.json
    # --model /data/object/liangxiong/double_iou/double_iou/kitti/faster_rcnn_53_3257.pth \
    # --lr 1e-5
# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net three_iou_org_ohem \
    # --out_path /data/object/liangxiong/three_iou_org_ohem \
    # --config configs/org_three_iou_kitti_config.json

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net three_iou_org_ohem_second \
    # --out_path /data/object/liangxiong/three_iou_best_second \
    # --config configs/org_three_iou_kitti_config.json \
    # --model /data/object/liangxiong/double_iou_new/double_iou/kitti/faster_rcnn_100_3257.pth
# --checkpoint 3257 \
    # --checkepoch 3 \
    # --r True
# --model /data/object/liangxiong/three_iou_best/three_iou_org_ohem/kitti/faster_rcnn_84_3257.pth
# --model /data/object/liangxiong/three_iou_/three_iou/kitti/faster_rcnn_39_3257.pth
# --checkpoint 3257 \
    # --checkepoch 7 \
    # --r True
# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net three_iou_org \
    # --out_path /data/object/liangxiong/three_iou_org \
    # --config configs/org_three_iou_kitti_config.json \
    # --checkepoch 23 \
    # --checkpoint 3257 \
    # --r True

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net three_iou_org_ohem \
    # --out_path /data/object/liangxiong/three_iou_attention \
    # --config configs/org_three_iou_kitti_config.json
# --checkepoch 44 \
    # --checkpoint 3257 \
    # --r True

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net cascade \
    # --out_path /data/object/liangxiong/cascade \
    # --config configs/cascade_kitti_config.json

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net three_iou_org \
    # --out_path /data/object/liangxiong/three_iou_org_ohem \
    # --config configs/org_three_iou_kitti_config.json

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net double_iou_slow \
    # --out_path /data/object/liangxiong/double_iou_slow_ohem \
    # --config configs/double_iou_kitti_config.json

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net double_iou_slow \
    # --out_path /data/object/liangxiong/double_iou_slow \
    # --config configs/double_iou_kitti_config.json \
    # --checkpoint 3257 \
    # --checkepoch 29 \
    # --r True

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net double_iou_slow \
    # --out_path /data/object/liangxiong/double_iou_slow_ohem_better \
    # --config configs/double_iou_kitti_config.json

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net double_iou_slow \
    # --out_path /data/object/liangxiong/double_iou_slow_01 \
    # --config configs/double_iou_kitti_config.json \
    # --checkpoint 3257 \
    # --checkepoch 17 \
    # --r True

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net single_iou \
    # --out_path /data/object/liangxiong/single_iou \
    # --config configs/refine_kitti_config.json \
    # --checkpoint 3257 \
    # --checkepoch 7 \
    # --r True

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net semantic \
    # --out_path /data/object/liangxiong/part07 \
    # --config configs/refine_kitti_config.json \
    # --model /data/object/liangxiong/semantic/semantic/kitti/faster_rcnn_100_3257.pth
# --model /data/object/liangxiong/part05/semantic/kitti/faster_rcnn_24_3257.pth
# --checkpoint 3257 \
    # --checkepoch 25 \
    # --r True

# --model /data/object/liangxiong/faster_rcnn/faster_rcnn/kitti/faster_rcnn_53_3257.pth

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net semantic \
    # --out_path /data/object/liangxiong/semantic_self \
    # --config configs/refine_kitti_config.json
# --checkpoint 3257 \
    # --checkepoch 130 \
    # --r True \
    # --lr 0.5

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net new_semantic \
    # --out_path /data/object/liangxiong/semantic_new \
    # --config configs/refine_kitti_config.json \
    # --checkpoint 3257 \
    # --checkepoch 13 \
    # --r True

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net semantic \
    # --out_path /data/object/liangxiong/semantic_anchors \
    # --config configs/refine_kitti_config.json \
    # --checkpoint 3257 \
    # --checkepoch 47 \
    # --r True

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net new_faster_rcnn \
    # --out_path /data/object/liangxiong/exp_iouweights_hem_great \
    # --config configs/kitti_config.json
# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net loss \
    # --out_path /data/object/liangxiong/loss \
    # --config configs/refine_kitti_config.json

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net LED \
    # --out_path /data/object/liangxiong/LED_clip \
    # --config configs/LED_kitti_config.json \
    # --model /data/object/liangxiong/faster_rcnn/faster_rcnn/kitti/faster_rcnn_53_3257.pth \
    # --lr 1e-3

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net iou_faster_rcnn \
    # --out_path /data/object/liangxiong/iou_exp \
    # --config configs/iou_kitti_config.json
# --model /data/object/liangxiong/semantic/semantic/kitti/faster_rcnn_24_3257.pth
# --checkpoint 3257 \
    # --checkepoch 5 \
    # --r True

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    # --net overlaps \
    # --out_path /data/object/liangxiong/overlaps \
    # --config configs/overlaps_kitti_config.json \
    # --checkpoint 3257 \
    # --checkepoch 10 \
    # --r True
