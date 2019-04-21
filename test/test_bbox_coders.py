# -*- coding: utf-8 -*-
import sys
sys.path.append('./')
from core import constants
from data import datasets
import bbox_coders
import torch


def build_dataset():
    dataset_config = {
        "type": "mono_3d_kitti",
        "classes": ["Car", "Pedestrian"],
        "cache_bev": False,
        "dataset_file": "data/demo.txt",
        "root_path": "/data"
    }

    dataset = datasets.build(dataset_config, transform=None, training=True)

    return dataset


def test_bbox_coders():
    coder_config = {'type': constants.KEY_ORIENTS}
    bbox_coder = bbox_coders.build(coder_config)

    dataset = build_dataset()
    sample = dataset[11]
    label_boxes_3d = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_3D])
    p2 = torch.from_numpy(sample[constants.KEY_STEREO_CALIB_P2])
    proposals = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_2D])
    num_instances = torch.from_numpy(sample[constants.KEY_NUM_INSTANCES])

    label_boxes_3d = torch.stack(3 * [label_boxes_3d[:num_instances]], dim=0)
    proposals = torch.stack(3 * [proposals[:num_instances]], dim=0)
    p2 = torch.stack(3 * [p2], dim=0)
    orients = bbox_coder.encode_batch(label_boxes_3d, proposals, p2)
    # import ipdb
    # ipdb.set_trace()
    print(orients.shape)
    encoded_cls_orients = torch.zeros_like(orients[:, :, :2])
    cls_orients = orients[:, :, :1].long()
    row = torch.arange(0, cls_orients.numel()).type_as(cls_orients)
    encoded_cls_orients.view(-1, 2)[row, cls_orients.view(-1)] = 1
    encoded_orients = torch.cat([encoded_cls_orients, orients[:, :, 1:]],
                                dim=-1)

    ry = bbox_coder.decode_batch(encoded_orients, proposals, proposals, p2)
    # import ipdb
    # ipdb.set_trace()
    print(ry)
    print(label_boxes_3d[:, :, -1])
    print(sample[constants.KEY_IMAGE_PATH])


def test_orient_coder():
    coder_config = {'type': constants.KEY_ORIENTS}
    bbox_coder = bbox_coders.build(coder_config)

    dataset = build_dataset()
    sample = dataset[0]
    label_boxes_3d = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_3D])
    p2 = torch.from_numpy(sample[constants.KEY_STEREO_CALIB_P2])
    proposals = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_2D])

    orients = bbox_coder.encode_batch(label_boxes_3d, proposals, p2)
    print(orients.shape)


def test_orientv3_coder():
    coder_config = {'type': constants.KEY_ORIENTS_V3}
    orient_coder = bbox_coders.build(coder_config)

    dataset = build_dataset()
    sample = dataset[0]
    label_boxes_3d = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_3D])
    num_instances = torch.from_numpy(sample[constants.KEY_NUM_INSTANCES])

    label_boxes_3d = torch.stack(3 * [label_boxes_3d[:num_instances]], dim=0)
    orients = orient_coder.encode_batch(label_boxes_3d)
    print(orients)


def test_orientv2_coder():
    coder_config = {'type': constants.KEY_ORIENTS_V2}
    bbox_coder = bbox_coders.build(coder_config)

    dataset = build_dataset()
    sample = dataset[0]
    label_boxes_3d = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_3D])
    p2 = torch.from_numpy(sample[constants.KEY_STEREO_CALIB_P2])
    proposals = torch.from_numpy(sample[constants.KEY_LABEL_BOXES_2D])
    num_instances = torch.from_numpy(sample[constants.KEY_NUM_INSTANCES])

    label_boxes_3d = torch.stack(1 * [label_boxes_3d[:num_instances]], dim=0)
    proposals = torch.stack(1 * [proposals[:num_instances]], dim=0)
    p2 = torch.stack(1 * [p2], dim=0)
    # import ipdb
    # ipdb.set_trace()
    orients = bbox_coder.encode_batch(label_boxes_3d, proposals, p2)
    print(orients)

    encoded_cls_orients = torch.zeros_like(orients[:, :, :3])
    cls_orients = orients[:, :, :1].long()
    row = torch.arange(0, cls_orients.numel()).type_as(cls_orients)
    encoded_cls_orients.view(-1, 3)[row, cls_orients.view(-1)] = 1
    encoded_orients = torch.cat([encoded_cls_orients, orients[:, :, 1:]],
                                dim=-1)

    ry = bbox_coder.decode_batch(encoded_orients, proposals, p2)
    print(ry)
    print(label_boxes_3d[:, :, -1])
    print(sample[constants.KEY_IMAGE_PATH])


if __name__ == '__main__':
    # test_bbox_coders()
    # test_orient_coder()
    # test_orientv3_coder()
    test_orientv2_coder()
