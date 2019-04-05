# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from core import constants


def align_stack(items, default_value=0):
    """
    items in list may be not aligned
    Args:
    items: list
    """
    num_max = 0
    for item in items:
        if num_max < items.shape[0]:
            num_max = items.shape[0]

    target_shape = [num_max]
    for i in range(1, len(items[0].shape)):
        target_shape.append(items[0].shape[i])
    target = torch.zeros(target_shape).type_as(items[0])
    targets = []
    for batch_ind, item in enumerate(items):
        target[:item.shape[0]] = item
        targets.append(target)

    return torch.stack(targets, dim=0)


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        # print(len(batch))
        batch_size = len(batch)
        # init dict
        all_keys = {}
        for key in batch[0]:
            all_keys[key] = []

        # copy to  all_keys
        for batch_ind in range(batch_size):
            sample = batch[batch_ind]
            for key in sample:
                all_keys[key].append(sample[key])
        image_infos = all_keys[constants.KEY_IMAGE_INFO]
        image_infos = [
            torch.from_numpy(image_info) for image_info in image_infos
        ]
        results = {}
        results[constants.KEY_IMAGE] = torch.stack(
            all_keys[constants.KEY_IMAGE], dim=0)
        results[constants.KEY_IMAGE_INFO] = torch.stack(image_infos, dim=0)
        print(all_keys[constants.KEY_LABEL_CLASSES])
        results[constants.KEY_LABEL_CLASSES] = align_stack(
            all_keys[constants.KEY_LABEL_CLASSES])
        results[constants.KEY_LABEL_BOXES_2D] = align_stack(
            all_keys[constants.KEY_LABEL_BOXES_2D])
        return results
        # for key in all_keys:

    # try:
    # all_keys[key] = torch.stack(all_keys[key], dim=0)
    # except:
    # all_keys[key] = align_stack(all_keys[key])
    # return all_keys
    # transposed_batch = list(zip(*batch))
    # import ipdb
    # ipdb.set_trace()
    # images = transposed_batch[0]
    # print(images)
    # targets = transposed_batch[1]
    # img_ids = transposed_batch[2]
    # return images, targets, img_ids
