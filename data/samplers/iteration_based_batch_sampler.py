# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch.utils.data.sampler import BatchSampler


class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler):
        self.batch_sampler = batch_sampler
        # self.num_iterations = num_iterations

    def __getattr__(self, name):
        return getattr(self.batch_sampler, name)

    def __iter__(self):
        # iteration = 1
        while True:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            # if hasattr(self.batch_sampler.sampler, "set_epoch"):
                # self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                # iteration += 1
                # if iteration > self.num_iterations:
                    # break
                yield batch

    def __len__(self):
        return self.num_iterations
