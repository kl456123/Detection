#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python new_train.py --cuda \
    --net faster_rcnn \
    --out_path ./experiments/new \
    --config configs/new_config.json
