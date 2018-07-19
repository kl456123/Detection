# -*- coding: utf-8 -*-

model_config = {
    'net': 'resnet50',
    'num_classes': 2,
    'output_stride': [8., 16., 32., 64., 128., 192., 384.],
    'input_shape': (384, 1300),
    'class_agnostic': True,
    'pretrained': True,
    'img_channels': 3,
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
        'dataset_file': 'train.txt'
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
    'save_dir': 'weights',
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
