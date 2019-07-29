#!/usr/bin/env bash


NET=geometry_v2

CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    --net ${NET} \
    --out_path /data/object/liangxiong/test \
    --config configs/${NET}.json
