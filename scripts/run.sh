#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 python train.py --cuda \
    --net faster_rcnn \
    --out_path /data/object/liangxiong/new \
    --config configs/new_config.json \
    --mGPUs
