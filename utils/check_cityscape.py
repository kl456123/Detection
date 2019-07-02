# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from PIL import Image

root_path = '/data/Cityscape'

phase = 'train'

# label_dir =
# data_dir =


def parse_data_path(abs_data_path, label_dir='./gtFine'):
    addr_name, id1, id2, cat = abs_data_path.split('_')
    label_suffix = 'gtFine_instanceIds.png'
    # label_suffix = 'gtFine_labelIds.png'
    label_fn = '_'.join([addr_name, id1, id2, label_suffix])
    label_dir = os.path.join(root_path, label_dir, phase)
    abs_label_dir = os.path.join(label_dir, addr_name)
    return os.path.join(abs_label_dir, label_fn)


def get_data(data_dir='leftImg8bit_trainvaltest/leftImg8bit/', phase=phase):
    data_dir = os.path.join(root_path, data_dir, phase)
    for addr_name in os.listdir(data_dir):
        abs_data_dir = os.path.join(data_dir, addr_name)
        for image_fn in os.listdir(abs_data_dir):
            abs_data_path = os.path.join(abs_data_dir, image_fn)
            image = cv2.imread(abs_data_path)
            abs_label_path = parse_data_path(image_fn)
            yield image, abs_label_path


'ulm_000007_000019_leftImg8bit.png'
'ulm_000007_000019_gtFine_instanceIds.png'


def get_label(label_dir='./gtFine', phase=phase):
    label_dir = os.path.join(root_path, label_dir, phase)
    for addr_name in os.listdir(label_dir):
        abs_data_dir = os.path.join(label_dir, addr_name)
        for image_fn in os.listdir(abs_data_dir):
            if 'instance' not in image_fn:
                continue
            abs_data_path = os.path.join(abs_data_dir, image_fn)
            image = cv2.imread(abs_data_path)
            yield image


def get_label_from_abs_path(abs_label_path):
    pass


def main():
    import ipdb
    ipdb.set_trace()
    for image, abs_label_path in get_data():
        label_image = Image.open(abs_label_path)
        label = np.array(label_image)
        # label = cv2.imread(abs_label_path)
        np.unique(label).shape


if __name__ == '__main__':
    main()
