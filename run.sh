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

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
# --net faster_rcnn \
# --out_path /data/object/liangxiong/use_iou \
# --config configs/refine_kitti_config.json \
# --model /data/object/liangxiong/faster_rcnn/faster_rcnn/kitti/faster_rcnn_53_3257.pth

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
    # --net semantic \
    # --out_path /data/object/liangxiong/semantic_weights \
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

CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
--net iou_faster_rcnn \
--out_path /data/object/liangxiong/iou_exp \
--config configs/iou_kitti_config.json \
--model /data/object/liangxiong/semantic/semantic/kitti/faster_rcnn_24_3257.pth
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
