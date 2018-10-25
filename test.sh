#!/bin/bash
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

CUDA_VISIBLE_DEVICES=1 python test_net.py --cuda \
    --checkpoint 3257 \
    --checkepoch 103 \
    --net semantic \
    --load_dir /data/object/liangxiong/semantic
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
    # --checkepoch 48 \
    # --net double_iou_second \
    # --load_dir /data/object/liangxiong/double_iou_second
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
