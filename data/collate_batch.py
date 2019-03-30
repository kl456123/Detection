# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        pass
        # transposed_batch = list(zip(*batch))
        # import ipdb
        # ipdb.set_trace()
        # images = transposed_batch[0]
        # print(images)
        # targets = transposed_batch[1]
        # img_ids = transposed_batch[2]
        # return images, targets, img_ids

