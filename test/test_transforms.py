# -*- coding: utf-8 -*-
from data import datasets
from data import transforms


def test_kitti_dataset():
    dataset_config = {
        'type': 'kitti',
        'root_path': '/data',
        'data_path': 'object/training/image_2',
        'label_path': 'object/training/label_2',
        'classes': ['Car', 'Pedestrian', 'Truck'],
        'dataset_file': './data/train.txt'
    }
    transforms_config = [{
        "type": "to_pil"
    }, {
        "type": "random_hsv"
    }, {
        "type": "random_brightness"
    }, {
        "type": "random_horizontal_flip"
    }, {
        "type": "random_zoomout"
    }, {
        "type": "fix_shape_resize",
        "size": [384, 1280]
    }]
    transform = transforms.build(transforms_config)

    dataset = datasets.build(
        dataset_config, transform=transform, training=True)

    #  dataset.visuliaze_sample(sample)

    for ind, sample in enumerate(dataset):
        print(ind)
        sample = dataset[5]
        dataset.visualize_samplev1(sample)


if __name__ == '__main__':
    test_kitti_dataset()
