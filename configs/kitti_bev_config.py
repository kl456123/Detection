# -*- coding: utf-8 -*-

# MODEL_CONFIG = {
# 'num_anchors': 6,
# # 'default_ratio': [0.0449, 0.0772, 0.115, 0.164, 0.227],
# 'default_ratio': [0.034, 0.034, 0.034],
# 'aspect_ratio': ((2.39, ), (2.39, )),
# }

model_config = {
    'net': 'resnet50',
    'num_classes': 2,
    'output_stride': [4., 8.],
    'input_shape': (800, 700),
    'class_agnostic': True,
    'pretrained': False,
    'img_channels': 6,
    'classes': ['bg', 'Car'],
    'rpn_config': {
        'din': 1024,
        'anchor_ratios': [0.5, 1, 2],
        'anchor_scales': [2, 3, 4],
        'feat_stride': 16,
        'pre_nms_topN': 12000,
        'post_nms_topN': 2000,
        'nms_thresh': 0.7,
        'min_size': 16,
        'rpn_clobber_positives': False,
        'rpn_negative_overlap': 0.3,
        'rpn_positive_overlap': 0.5,
        'rpn_batch_size': 512,
        'rpn_fg_fraction': 0.5,
        'rpn_bbox_inside_weights': [1.0, 1.0, 1.0, 1.0],
        'rpn_positive_weight': -1.0,
    },
    'pooling_size': 7,
    'pooling_mode': 'align',
    'crop_resize_with_max_pool': False,
    'truncated': False,
    'proposal_target_layer_config': {
        'nclasses': 2,
        'bbox_normalize_means': (0.0, 0.0, 0.0, 0.0),
        'bbox_normalize_stds': (0.1, 0.1, 0.2, 0.2),
        'bbox_inside_weights': [1.0, 1.0, 1.0, 1.0],
        'batch_size': 512,
        'fg_fraction': 0.25,
        'bbox_normalize_targets_precomputed': True,
        'fg_thresh': 0.5,
        'bg_thresh': 0.5,
        'bg_thresh_lo': 0.0,
    },
}

data_config = {
    'name': 'kitti',
    'dataset_config': {
        'root_path': '/data/object/training',
        'dataset_file': 'train.txt',
        # cache bev map to pkl file
        'cache_bev': False,
        'cache_dir': '/data/object/training/cache_bev',
        'bev_config': {
            'height_lo': -0.2,
            'height_hi': 2.3,
            'num_slices': 5,
            'voxel_size': 0.1,
            'area_extents':
            [[-40, 40], [-5, 3],
             [0, 70]],  # [[xmin, xmax], [ymin, ymax], [zmin, zmax]]
            'density_threshold': 1,
        },
        'camera_baseline': 0.54
    },
    'transform_config': {
        'normal_mean': [0.485, 0.456, 0.406],
        'normal_van': [0.229, 0.224, 0.225],
        'resize_range': [0.2, 0.4],
        'random_brightness': 10,
        'crop_size': (284, 1300),
        'random_blur': 0,
    },
    'dataloader_config': {
        'shuffle': True,
        'batch_size': 1,
        'num_workers': 1
    }
}

eval_data_config = {
    'name': 'kitti',
    'dataset_config': {
        'root_path': '/data/object/training',
        'dataset_file': 'val.txt'
    },
    'transform_config': {
        'normal_mean': [0.485, 0.456, 0.406],
        'normal_van': [0.229, 0.224, 0.225],
        # 'resize_range': [0.2, 0.4],
        # 'random_brightness': 10,
        # 'crop_size': (284, 1300),
        # 'random_blur': 0,
    },
    'dataloader_config': {
        'shuffle': True,
        'batch_size': 1,
        'num_workers': 1
    },
}

train_config = {
    'rng_seed': 3,
    'save_dir': '/data/object/liangxiong/bev',
    'device_ids': [0],
    'disp_interval': 100,
    'max_epochs': 100,
    'checkpoint_interval': 10000,
    'mGPUs': True,
    'clip_gradient': 10,
    'start_epoch': 1,
    'scheduler_config': {
        'type': 'step',
        'lr_decay_gamma': 0.1,
        'lr_decay_step': 20,
        'last_epoch': -1,
    },
    'optimizer_config': {
        'type': 'sgd',
        'momentum': 0.9,
        'lr': 0.001,
    }
}

eval_config = {
    # used for testing one image
    'demo_file': '',
    'checkpoint': 3257,
    'checkepoch': 100,
    'rng_seed': 3,
    'load_dir': '/data/liangxiong/models',
    'max_per_image': 100,
    'bbox_reg': True,
    'bbox_normalize_targets_precomputed': False,
    'bbox_normalize_means': [],
    'bbox_normalize_stds': [],
    'batch_size': 1,
    'class_agnostic': True,
    'thresh': 0.5,
    'nms': 0.3,
    'classes': ['bg', 'Car'],
    'eval_out': './results/data',
}
