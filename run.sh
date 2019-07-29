#!/usr/bin/env bash


# uncomment the one as you want
# NET=geometry_v2
# NET=geometry_v1
NET=geometry_v3

CUDA_VISIBLE_DEVICES=0 python trainval_net.py --cuda \
    --net ${NET} \
    --out_path /data/object/liangxiong/test \
    --config configs/${NET}.json
