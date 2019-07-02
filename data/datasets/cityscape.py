import os
from PIL import Image
from data.det_dataset import DetDataset
import numpy as np
import random
import cv2

# wavedata for kitti
from wavedata.tools.core import calib_utils
from wavedata.tools.obj_detection import obj_utils
from cityscapesscripts.evaluation import instance
from cityscapesscripts.helpers.csHelpers import *

from core import constants
from utils.registry import DATASETS


@DATASETS.register('cityscape')
class CityScapeDataset(DetDataset):
    def __init__(self, config, transform=None, training=True, logger=None):
        super().__init__(training)
        # root path of dataset
        self._root_path = config['root_path']
        self._label_path = config['label_path']
        self._data_path = config['data_path']
        if training:
            phase = 'train'
        else:
            phase = 'val'
        self.phase = phase

        self.transforms = transform

        # classes to be trained
        # 0 refers to bg
        classes = ['bg']
        self.classes = classes + config['classes']
        sample_names = self._generate_sample_names(phase)

        # import ipdb
        # ipdb.set_trace()
        # if self.training:
        # self.sample_names = sorted(self.filter_sample_names(sample_names))
        # else:
        self.sample_names = sorted(sample_names)

        self.max_num_boxes = 100

    def filter_sample_names(self, sample_names):
        loaded_sample_names = []
        for image_path in sample_names:
            # sample_name = self.get_sample_name_from_path(sample_path)
            abs_label_path = self.parse_data_path(image_path)
            label_image = np.array(Image.open(abs_label_path))
            label_boxes_2d, label_classes, label_instance_mask = self.parse_label_image(
                label_image)
            if label_boxes_2d.shape[0] > 0:
                loaded_sample_names.append(image_path)

        return loaded_sample_names

    def _generate_sample_names(self, phase):
        image_dir = os.path.join(self._root_path, self._data_path, phase)
        sample_names = []
        for attr_name in os.listdir(image_dir):
            abs_image_dir = os.path.join(image_dir, attr_name)
            sample_names.append(
                self.load_sample_names_from_image_dir(abs_image_dir))
        return np.concatenate(sample_names, axis=0)

    def getLabelID(self, instID):
        if (instID < 1000):
            return instID
        else:
            return int(instID / 1000)

    def _class_str_to_index(self, class_type):
        return self.classes.index(class_type)

    def parse_label_image(self, label_image):
        # import ipdb
        # ipdb.set_trace()
        total_index = 0
        label_boxes_2d = []
        label_classes = []
        label_instance_mask = np.zeros_like(label_image)
        for instanceId in np.unique(label_image):
            if instanceId < 1000:
                continue
            label_id = self.getLabelID(instanceId)
            label = id2label[label_id]
            if label.hasInstances and label.name in self.classes:
                total_index += 1
                instance_coords = np.where(label_image == instanceId)
                xmin = instance_coords[1].min()
                xmax = instance_coords[1].max()
                ymin = instance_coords[0].min()
                ymax = instance_coords[0].max()
                label_instance_mask[instance_coords] = total_index
                label_boxes_2d.append([xmin, ymin, xmax, ymax])
                label_classes.append(self._class_str_to_index(label.name))

        return np.asarray(label_boxes_2d), np.asarray(
            label_classes), np.asarray(label_instance_mask)[None]

    def parse_data_path(self, abs_data_path):
        addr_name, id1, id2, cat = os.path.basename(abs_data_path).split('_')
        label_suffix = 'gtFine_instanceIds.png'
        # label_suffix = 'gtFine_labelIds.png'
        label_fn = '_'.join([addr_name, id1, id2, label_suffix])
        label_dir = os.path.join(self._root_path, self._label_path, self.phase)
        abs_label_dir = os.path.join(label_dir, addr_name)
        return os.path.join(abs_label_dir, label_fn)

    def get_random_index(self):
        index = random.random() * len(self)
        return int(index)

    def get_training_sample(self, index):
        # import ipdb
        # ipdb.set_trace()
        # loop until find one
        while True:
            image_path = self.sample_names[index]

            # image
            image_input = Image.open(image_path)
            image_shape = image_input.size[::-1]
            # no scale now
            image_scale = (1.0, 1.0)
            image_info = image_shape + image_scale

            # label
            abs_label_path = self.parse_data_path(image_path)
            label_image = np.array(Image.open(abs_label_path))
            label_boxes_2d, label_classes, label_instance_mask = self.parse_label_image(
                label_image)
            if label_boxes_2d.shape[0] > 0:
                break
            else:
                index = self.get_random_index()

        transform_sample = {}
        transform_sample[constants.KEY_IMAGE] = image_input
        transform_sample[constants.KEY_LABEL_BOXES_2D] = label_boxes_2d.astype(
            np.float32)
        transform_sample[
            constants.KEY_LABEL_INSTANCES_MASK] = label_instance_mask.astype(
                np.int64)
        transform_sample[constants.KEY_LABEL_CLASSES] = label_classes.astype(
            np.int64)
        transform_sample[constants.KEY_IMAGE_PATH] = image_path

        # (h,w,scale)
        transform_sample[constants.KEY_IMAGE_INFO] = np.asarray(
            image_info, dtype=np.float32)

        #  import ipdb
        #  ipdb.set_trace()

        return transform_sample

    def pad_sample(self, sample):
        label_boxes_2d = sample[constants.KEY_LABEL_BOXES_2D]
        label_classes = sample[constants.KEY_LABEL_CLASSES]
        all_label_boxes_2d = np.zeros((self.max_num_boxes,
                                       label_boxes_2d.shape[1]))
        all_label_classes = np.zeros((self.max_num_boxes, ))
        # assign it with bg label
        all_label_classes[...] = 0
        num_boxes = label_boxes_2d.shape[0]
        all_label_classes[:num_boxes] = label_classes
        all_label_boxes_2d[:num_boxes] = label_boxes_2d

        sample[constants.KEY_NUM_INSTANCES] = np.asarray(
            num_boxes, dtype=np.int32)

        sample[constants.KEY_LABEL_BOXES_2D] = all_label_boxes_2d.astype(
            np.float32)
        sample[constants.KEY_LABEL_CLASSES] = all_label_classes

        # resize instance mask according to image info
        # image_info = sample[constants.KEY_IMAGE_INFO]
        # label_instance_mask = sample[constants.KEY_LABEL_INSTANCES_MASK].astype(np.uint8)
        # h, w = image_info[:2]
        # import ipdb
        # ipdb.set_trace()
        # tmp = label_instance_mask[0, :, :, None].repeat(3, axis=-1)
        # label_instance_mask = cv2.resize(
            # tmp, dsize=(w, h), interpolation=cv2.INTER_CUBIC)

        # sample[constants.KEY_LABEL_INSTANCES_MASK] = label_instance_mask.astype(np.int64)

        return sample

    def get_testing_sample(self, index):
        pass

    def visuliaze_sample(self, sample):
        # import ipdb
        # ipdb.set_trace()
        import matplotlib.pyplot as plt
        image = sample[constants.KEY_IMAGE]
        num_instances = sample[constants.KEY_NUM_INSTANCES]
        label_boxes_2d = sample[constants.KEY_LABEL_BOXES_2D].astype(np.int32)
        label_instance_mask = sample[constants.KEY_LABEL_INSTANCES_MASK]
        instance_mask = np.zeros_like(label_instance_mask)
        instance_box = label_boxes_2d[2]
        instance_mask[0, instance_box[1]:instance_box[3], instance_box[0]:
                      instance_box[2]] = label_instance_mask[0, instance_box[
                          1]:instance_box[3], instance_box[0]:instance_box[2]]
        plt.imshow(instance_mask[0])
        plt.show()

        from utils.visualize import visualize_bbox
        image = np.asarray(image)
        visualize_bbox(image, label_boxes_2d[:num_instances], save=True)


if __name__ == '__main__':
    from utils.drawer import ImageVisualizer
    image_dir = '/data/object/training/image_2'
    result_dir = './results/data'
    save_dir = 'results/images'
    calib_dir = '/data/object/training/calib'
    label_dir = None
    calib_file = None
    visualizer = ImageVisualizer(
        image_dir,
        result_dir,
        label_dir=label_dir,
        calib_dir=calib_dir,
        calib_file=calib_file,
        online=False,
        save_dir=save_dir)
    dataset_config = {
        "classes": ["car"],
        "data_path": "leftImg8bit_trainvaltest/leftImg8bit",
        "dataset_file": "data/demo.txt",
        "label_path": "./gtFine",
        "root_path": "/data/Cityscape",
        "type": "cityscape"
    }
    # import ipdb
    # ipdb.set_trace()
    transform = None
    dataset = CityScapeDataset(dataset_config, transform, training=True)
    # for sample in dataset:
    # sample = dataset[4]
    # visualizer =
    print(dataset[1][constants.KEY_IMAGE_PATH])
    dataset.visuliaze_sample(dataset[1])
