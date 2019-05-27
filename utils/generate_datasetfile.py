# -*- coding: utf-8 -*-

import os

TRAINING_IMAGE_DIR = '/data/nuscenes_kitti/train/image_2'
TESTING_IMAGE_DIR = '/data/nuscenes_kitti/val/image_2'


def generate_dataset_file(image_dir, dataset_fn):
    sample_names = []
    for file in os.listdir(image_dir):
        sample_name = os.path.splitext(file)[0]
        sample_names.append(sample_name)

    with open(dataset_fn, 'w') as f:
        f.write('\n'.join(sample_names))


generate_dataset_file(TRAINING_IMAGE_DIR, 'train.txt')
generate_dataset_file(TESTING_IMAGE_DIR, 'val.txt')
