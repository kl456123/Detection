# -*- coding: utf-8 -*-
import sys
sys.path.append('./')
from core import constants
from data import datasets
import bbox_coders
import torch



def test_bbox_coders():
    coder_config = {'type': constants.KEY_ORIENTS}
    bbox_coder = bbox_coders.build(coder_config)
    dataset_config = {
        "type": "mono_3d_kitti",
        "classes": ["Car", "Pedestrian"],
        "cache_bev": False,
        "dataset_file": "data/train.txt",
        "root_path": "/home/breakpoint/Data/KITTI"
    }

    dataset = datasets.build(dataset_config, transform=None, training=False)
    sample = dataset[0]
    label_boxes_3d = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_3D])
    p2 = torch.from_numpy(sample[constants.KEY_STEREO_CALIB_P2])
    proposals = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_2D])

    label_boxes_3d = torch.stack(3*[label_boxes_3d], dim=0)
    proposals = torch.stack(3*[proposals], dim=0)
    p2 = torch.stack(3*[p2], dim=0)
    orients = bbox_coder.encode_batch(label_boxes_3d, proposals, p2)
    print(orients.shape)


if __name__ == '__main__':
    test_bbox_coders()
