# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
import torch
from model.nms.nms_gpu import nms_gpu

def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""
    if dets.shape[0] == 0:
        return []
    # ---numpy version---
    # ---pytorch version---
    return nms_gpu(dets, thresh)
