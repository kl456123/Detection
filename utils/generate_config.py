# -*- coding: utf-8 -*-
# SWITCH that you only should care about
# DATASET_TYPE = 'mono_3d_kitti'
DATASET_TYPE = 'nuscenes'
# DATASET_TYPE = 'bdd'
# NET_TYPE = 'fpn_corners_2d'
NET_TYPE = 'fpn'
# NET_TYPE = 'fpn_mono_3d_better'
JOBS = True
DEBUG = False

# enable debug mode
if DEBUG:
    training_batch_size = 1
    num_workers = 1
    base_lr = 0.001
    num_iters = 1000
    checkpoint_interval = 100
    disp_interval = 100
    training_dataset_file = "data/demo.txt"
    testing_dataset_file = "data/demo.txt"
    if DATASET_TYPE == 'nuscenes_kitti':
        training_dataset_file = "data/nuscenes_demo.txt"
        testing_dataset_file = "data/nuscenes_demo.txt"
else:
    training_batch_size = 32
    num_workers = 48
    base_lr = 0.02
    num_iters = 600000
    checkpoint_interval = 4000
    disp_interval = 800
    training_dataset_file = "data/train.txt"
    testing_dataset_file = "data/val.txt"
    if DATASET_TYPE == 'nuscenes_kitti':
        training_dataset_file = 'data/nuscenes_train.txt'
        testing_dataset_file = 'data/nuscenes_val.txt'

# common config
testing_batch_size = 1
normal_mean = [0.485, 0.456, 0.406]
normal_std = [0.229, 0.224, 0.225]
training_transform_names = [
    "random_hsv", "random_brightness", "random_horizontal_flip",
    "fix_shape_resize", "to_tensor", "normalize"
]
testing_transform_names = ["fix_shape_resize", "to_tensor", "normalize"]
testing_nms = 0.3
training_nms = 0.7
testing_thresh = 0.5
class_agnostic = True
use_focal_loss = False
post_nms_topN = 2000
pre_nms_topN = 12000
pooling_size = 7
pooling_mode = 'align'
feature_extractor_type = 'fpn'
net_arch = 'res18_pruned'
rpn_fg_fraction = 0.25
if net_arch == 'res18_pruned':
    ndin = [64, 128, 256, 512]
elif net_arch == 'res50':
    ndin = [256, 512, 1024, 2048]
rpn_ndin = 256
rpn_min_size = 16

if DATASET_TYPE in ['kitti', 'mono_3d_kitti', 'nuscenes_kitti']:
    # KITTI CONFIG
    root_path = '/data'
    classes = ['Car']
    if DATASET_TYPE == 'nuscenes_kitti':
        root_path = '/data/nuscenes_kitti'
        classes = [
            "bus", "bicycle", "car", "motorcycle", "truck", "trailer",
            "construction_vehicle"
        ]
    dataset_type = DATASET_TYPE
    image_size = [384, 1280]
    freeze_2d = True
    use_proj_2d = False
elif DATASET_TYPE == 'bdd':
    # BDD CONFIG
    root_path = '/data'
    classes = ['car']
    # classes = ["car", "person", "bus", "motor", "rider", "train", "truck"]
    dataset_type = 'bdd'
    training_dataset_file = "bdd100k_labels_images_train.json"
    testing_dataset_file = "bdd100k_labels_images_val.json"
    if JOBS:
        interval_str = ""
    else:
        interval_str = "100k"
    testing_data_path = "images/{}/val".format(interval_str)
    training_data_path = "images/{}/train".format(interval_str)

    if DEBUG:
        # the same file
        training_dataset_file = testing_dataset_file
        training_data_path = testing_data_path

    # if JOBS:
    # data_path = "images/train"
    # rpn_fg_fraction = 0.1

    root_path = "/data/bdd/bdd100k/"
    image_size = [384, 768]
elif DATASET_TYPE == "nuscenes":
    dataset_type = 'nuscenes'
    classes = [
        "bus", "bicycle", "car", "motorcycle", "truck", "trailer",
        "construction_vehicle", "pedestrian"
    ]
    root_path = "/data/nuscenes"
    training_dataset_file = "trainval.json"
    testing_dataset_file = training_dataset_file
    data_path = "samples/CAM_FRONT"
    label_path = "."
    image_size = [384, 1280]
else:
    raise TypeError('dataset type {} is unknown !'.format(DATASET_TYPE))


def generate_dataloader_config(training):
    if training:
        dataloader_config = {
            "batch_size": training_batch_size,
            "num_workers": num_workers,
            "shuffle": True
        }
    else:
        # use default config when testing
        dataloader_config = {
            "batch_size": testing_batch_size,
            "num_workers": 1,
            "shuffle": False
        }
    return dataloader_config


def generate_dataset_config(training):
    dataset_config = {
        "type": dataset_type,
        "classes": classes,
    }
    if training:
        dataset_config.update({
            "dataset_file": training_dataset_file,
            "root_path": root_path
        })
    else:
        dataset_config.update({
            "dataset_file": testing_dataset_file,
            "root_path": root_path
        })

    if dataset_type == 'kitti':
        # no need to add anything else
        pass
    elif dataset_type == 'mono_3d_kitti':
        dataset_config.update({'use_proj_2d': use_proj_2d})
    elif dataset_type == 'bdd':
        if training:

            dataset_config.update({
                "data_path": training_data_path,
                "label_path": "labels"
            })
        else:
            dataset_config.update({
                "data_path": testing_data_path,
                "label_path": "labels"
            })
    elif dataset_type == 'coco':
        pass
    elif dataset_type == 'nuscenes':
        dataset_config.update({"data_path": data_path, "label_path": "."})
    return dataset_config


def generate_transform_config(transform_names):
    assert isinstance(transform_names, list) or isinstance(
        transform_names, tuple)

    transform_config_maps = {
        "random_hsv": {
            "type": "random_hsv"
        },
        "random_brightness": {
            "type": "random_brightness"
        },
        "fix_shape_resize": {
            "type": "fix_shape_resize",
            "size": image_size
        },
        "random_horizontal_flip": {
            "type": "random_horizontal_flip"
        },
        "normalize": {
            "type": "normalize",
            "normal_mean": normal_mean,
            "normal_std": normal_std
        },
        "to_tensor": {
            "type": "to_tensor"
        }
    }
    transform_config = []
    for name in transform_names:
        transform_config.append(transform_config_maps[name])
    return transform_config


def generate_data_config(transform_names, training):
    transform_config = generate_transform_config(transform_names)
    dataloader_config = generate_dataloader_config(training)
    dataset_config = generate_dataset_config(training)

    data_config = {
        'transform_config': transform_config,
        'dataloader_config': dataloader_config,
        'dataset_config': dataset_config
    }
    return data_config


def generate_eval_config():
    eval_config = {
        "batch_size": testing_batch_size,
        "class_agnostic": class_agnostic,
        "classes": classes,
        "eval_out": "./results/data",
        "eval_out_anchors": "./results/anchors",
        "eval_out_rois": "./results/rois",
        "nms": testing_nms,
        "rng_seed": 3,
        "thresh": testing_thresh
    }
    return eval_config


def generate_model_config():
    anchor_config = {
        "use_pyramid": True,
        "type": "default",
        "anchor_offset": [0, 0],
        "anchor_stride": [16, 16],
        "aspect_ratios": [0.5, 0.8, 1],
        "base_anchor_size": 16,
        "scales": [2, 4, 8, 16]
    }
    feature_extractor_config = {
        "type": feature_extractor_type,
        "pretrained_path": "./data/pretrained_model",
        "pooling_size": pooling_size,
        "net_arch": net_arch,
        "pretrained": True,
        "ndin": ndin
    }
    rpn_target_generator_config = {
        "type": "faster_rcnn",
        "similarity_calc_config": {
            "type": "center"
        },
        "fg_thresh": 0.7,
        "bg_thresh": 0.3,
        "coder_config": {
            "type": "center",
        },
        "matcher_config": {
            "type": "bipartitle"
        },
        "sampler_config": {
            "num_samples": 1024,
            "type": "balanced",
            "fg_fraction": rpn_fg_fraction
        },
        "analyzer_config": {}
    }
    rcnn_target_generate_config = [{
        "type": "faster_rcnn",
        "similarity_calc_config": {
            "type": "center"
        },
        "fg_thresh": 0.5,
        "bg_thresh": 0.5,
        "coder_config": {
            "type": "corner",
        },
        "matcher_config": {
            "type": "argmax"
        },
        "sampler_config": {
            "num_samples": 512,
            "type": "balanced",
            "fg_fraction": 0.5
        },
        "analyzer_config": {}
    }]

    rpn_config = {
        "type": "fpn_rpn",
        "use_focal_loss": use_focal_loss,
        "anchor_generator_config": anchor_config,
        "target_generator_config": rpn_target_generator_config,
        "din": rpn_ndin,
        "min_size": rpn_min_size,
        "nms_thresh": training_nms,
        "post_nms_topN": post_nms_topN,
        "pre_nms_topN": pre_nms_topN
    }
    model_config = {
        "rpn_config": rpn_config,
        "class_agnostic": class_agnostic,
        "classes": classes,
        "target_generator_config": rcnn_target_generate_config,
        "feature_extractor_config": feature_extractor_config,
        "pooling_mode": pooling_mode,
        "pooling_size": pooling_size,
        "truncated": False,
        "use_focal_loss": use_focal_loss
    }
    if NET_TYPE == 'fpn_corners_2d':
        model_config['use_filter'] = True
    elif NET_TYPE == 'fpn_corners_3d':
        model_config['freeze_2d'] = freeze_2d

    return model_config


def generate_train_config():
    optimizer_config = {
        "base_lr": base_lr,
        "bias_lr_factor": 1,
        "weight_decay_bias": 0,
        "weight_decay": 0,
        "momentum": 0.9,
        "type": "sgd",
        "eps": 1e-8
    }
    scheduler_config = {
        "last_step": -1,
        "lr_decay_gamma": 0.1,
        "lr_decay_step": 60000,
        "milestones": [60000, 120000, 240000],
        "type": "multi_step",
        "warmup_method": "linear",
        "warmup_iters": 2000,
        "warmup_factor": 0.333
    }

    train_config = {
        "num_iters": num_iters,
        "checkpoint_interval": checkpoint_interval,
        "clip_gradient": 10,
        "disp_interval": disp_interval,
        "optimizer_config": optimizer_config,
        "rng_seed": 3,
        "scheduler_config": scheduler_config,
        "start_epoch": 1
    }
    return train_config


def generate_config():
    data_config = generate_data_config(training_transform_names, True)
    eval_data_config = generate_data_config(testing_transform_names, False)
    eval_config = generate_eval_config()

    model_config = generate_model_config()
    train_config = generate_train_config()
    config = {
        'data_config': data_config,
        'eval_data_config': eval_data_config,
        'eval_config': eval_config,
        'model_config': model_config,
        'train_config': train_config
    }
    import json
    net = NET_TYPE
    if DEBUG:
        json_file = 'configs/test_config.json'
    else:
        json_file = 'configs/{}_{}_config.json'.format(net, dataset_type)
    with open(json_file, 'w') as f:
        json.dump(config, f, indent=4, sort_keys=True)


def generate_bdd_config():
    pass


def main():
    generate_config()


if __name__ == '__main__':
    main()
