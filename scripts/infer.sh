#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python inference.py --cuda \
    --net geometry_v1 \
    --config configs/geometry_v1.json\
    --model /data/object/liangxiong/mono_3d_final_plus/mono_3d_final_plus/kitti/faster_rcnn_30_1518.pth
