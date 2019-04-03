#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda \
    --net faster_rcnn \
    --out_path /data/object/liangxiong/new \
    --config configs/new_config.json \
    --mGPUs
# --model /data/liangxiong/detection/data/pretrained_model/resnet50-19c8e357.pth
