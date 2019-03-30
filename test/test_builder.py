# -*- coding: utf-8 -*-
import sys
print(sys.path.append('.'))

import similarity_calcs
import torch
from data import transforms
import samplers
from models import detectors
import anchor_generators
import matchers
import bbox_coders
import unittest



class Test(unittest.TestCase):
    def test_xx(self):
        pass


def test_build_transform():
    transform_config = [{'type': 'to_tensor'}]
    transform = transforms.build(transform_config)
    print(transform)


def test_build_sampler():
    # subsample_batch and subsample function pass!!
    sampler_config = {'type': 'balanced', 'fg_fraction': 0.5}
    sampler = samplers.build(sampler_config)
    num_samplers = 100
    reg_weights = torch.rand((10, 2000))
    cls_weights = torch.ones_like(reg_weights)

    pos_indicator = reg_weights > 0.5
    indicator = cls_weights > 0
    # batch_weights = sampler.subsample(
    # num_samplers, pos_indicator, indicator=indicator)
    batch_weights = sampler.subsample_batch(
        num_samplers, pos_indicator, indicator=indicator)

    for i in range(batch_weights.shape[0]):
        weights = batch_weights[i]
        print(
            'Equal or not ({}/{})'.format(weights[weights].numel(), num_samplers))


def generate_anchors():
    anchor_generator_config = {
        "type": "default",
                "anchor_offset": [0, 0],
                "anchor_stride": [16, 16],
                "aspect_ratios": [0.5, 0.8, 1],
                "base_anchor_size": 16,
                "scales": [2, 4, 8, 16]
    }
    anchor_generator = anchor_generators.build(anchor_generator_config)
    feature_map_list = [(24, 80)]
    input_size = [384, 1280]
    anchors = anchor_generator.generate(feature_map_list, input_size)
    return anchors


def generate_fake_gt_boxes():
    gts = torch.tensor([100, 124, 125, 232]).float()
    gts.view(-1, 1, 4)
    return gts


def test_build_anchor_generator():
    anchors = generate_anchors()
    print(anchors[:10, ])
    print(anchors[:, 0].min())
    print(anchors[:, 1].min())
    print(anchors[:, 2].max())
    print(anchors[:, 3].max())


def test_target_assigner():
    pass


def test_analyzer():
    pass


def test_build_matcher():
    matcher_config = {"type": "argmax", "thresh": 0.5}
    matcher = matchers.build(matcher_config)

    match_quality_matrix = torch.rand((10, 1000, 5))
    match = matcher.match_batch(match_quality_matrix, thresh=0.5)
    print(match.shape)
    print(match[match == -1].shape)


def test_build_similarity_calc():
    similarity_calc_config = {'type': "center"}
    similarity_calc = similarity_calcs.build(similarity_calc_config)

    anchors = generate_anchors()
    gt_boxes = generate_fake_gt_boxes()
    match_quality_matrix_batch = similarity_calc.compare_batch(
        anchors, gt_boxes)
    print(match_quality_matrix_batch.shape)


def test_build_model():
    model_config = {
        'type': 'faster_rcnn',
        'num_stages': 2,
        'classes': ['Car', 'Truck'],
        'class_agnostic': False,
        'pooling_size': 7,
        'pooling_mode': 'roi_align',
        'use_focal_loss': True,
        'truncated': True,
        'batch_size': 1,
        "feature_extractor_config": {
            "type": "resnet",
            "pretrained_models_dir": "./data/pretrained_model",
            "net_arch": "res50",
            "separate_feat": False,
            "use_cascade": True,
            "class_agnostic": True,
            "classes": ["bg", "Car"],
            "img_channels": 3,
            "pretrained_model": "",
            "pretrained": True
        },
        "rpn_config": {
            "type": "rpn",
            "use_iou": False,
            "use_focal_loss": True,
            "anchor_generator_config": {
                "type": "default",
                "anchor_offset": [0, 0],
                "anchor_stride": [16, 16],
                "aspect_ratios": [0.5, 0.8, 1],
                "base_anchor_size": 16,
                "scales": [2, 4, 8, 16]
            },
            "din": 1024,
            "min_size": 16,
            "nms_thresh": 0.7,
            "post_nms_topN": 1000,
            "pre_nms_topN": 12000,
            "rpn_batch_size": 1024,
            "num_reg_samples": 1024,
            "num_cls_samples": 512,
            "sampler_config": {
                "type": "balanced",
                "fg_fraction": 0.25
            },
            "target_generator_config": [{
                "target_assigner_config": {
                    "type": "faster_rcnn",
                    "similarity_calc_config": {
                        "type": "center"
                    },
                    "fg_thresh": 0.3,
                    "bg_thresh": 0.3,
                    "coder_config": {
                        "type": "center",
                        "bbox_normalize_targets_precomputed": False
                    },
                    "matcher_config": {
                        "type": "bipartitle"
                    }
                },
                "sampler_config": {
                    "type": "balanced",
                    "fg_fraction": 0.5
                },
                "analyzer_config": {}
            }],
            "use_score": False
        },
        "target_generator_config": [{
            "target_assigner_config": {
                "type": "faster_rcnn",
                "similarity_calc_config": {
                    "type": "center"
                },
                "fg_thresh": 0.3,
                "bg_thresh": 0.3,
                "coder_config": {
                    "type": "center",
                    "bbox_normalize_targets_precomputed": False
                },
                "matcher_config": {
                    "type": "bipartitle"
                }
            },
            "sampler_config": {
                "type": "balanced",
                "fg_fraction": 0.5
            },
            "analyzer_config": {}
        }, {
            "target_assigner_config": {
                "type": "faster_rcnn",
                "similarity_calc_config": {
                    "type": "center"
                },
                "fg_thresh": 0.3,
                "bg_thresh": 0.3,
                "coder_config": {
                    "type": "center",
                    "bbox_normalize_targets_precomputed": False
                },
                "matcher_config": {
                    "type": "bipartitle"
                }
            },
            "sampler_config": {
                "type": "balanced",
                "fg_fraction": 0.5
            },
            "analyzer_config": {}
        }],
    }
    model = detectors.build(model_config)
    print(model)


if __name__ == '__main__':
    test_build_transform()
    test_build_sampler()
    test_build_anchor_generator()
    # test_build_model()
    test_build_matcher()
    test_build_similarity_calc()
