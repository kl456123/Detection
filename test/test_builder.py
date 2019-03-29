# -*- coding: utf-8 -*-

import sys
print(sys.path.append('.'))
from data import transforms
import samplers
from models import detectors


def test_build_transform():
    transform_config = [{'type': 'to_tensor'}]
    transform = transforms.build(transform_config)
    print(transform)


def test_build_sampler():
    sampler_config = {'type': 'balanced', 'fg_fraction': 0.5}
    sampler = samplers.build(sampler_config)
    print(sampler)


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
    test_build_model()
