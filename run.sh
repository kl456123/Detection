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

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
# --net faster_rcnn \
# --out_path /data/object/liangxiong/faster_rcnn_detection_all \
# --config configs/refine_kitti_config.json

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
# --net new_faster_rcnn \
# --out_path /data/object/liangxiong/exp_iouweights_hem_great \
# --config configs/kitti_config.json

# CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
# --net gate_faster_rcnn \
# --out_path /data/object/liangxiong/gate \
# --config configs/gate_kitti_config.json

CUDA_VISIBLE_DEVICES=1 python trainval_net.py --cuda \
--net iou_faster_rcnn \
--out_path /data/object/liangxiong/iou_exp \
--config configs/iou_kitti_config.json
