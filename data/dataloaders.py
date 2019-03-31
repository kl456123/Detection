# -*- coding: utf-8 -*-

import torch
from . import transforms
from . import datasets
from . import samplers
from .collate_batch import BatchCollator


def build(config):
    dataloader_config = config['dataloader_config']
    batch_size = dataloader_config['batch_size']
    shuffle = dataloader_config['shuffle']
    num_workers = dataloader_config['num_workers']

    # build transform first
    transform_config = config['transform_config']
    transform = transforms.build(transform_config)

    # then build dataset
    dataset_config = config['dataset_config']
    dataset = datasets.build(dataset_config, transform)
    # for debug
    dataset[0]

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers)
    return dataloader


def make_data_sampler(dataset, shuffle):
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(dataset,
                            sampler,
                            images_per_batch):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_batch, drop_last=False)
    batch_sampler = samplers.IterationBasedBatchSampler(
        batch_sampler)
    return batch_sampler


def make_data_loader(config):
    dataloader_config = config['dataloader_config']
    batch_size = dataloader_config['batch_size']
    shuffle = dataloader_config['shuffle']
    num_workers = dataloader_config['num_workers']

    # build transform first
    transform_config = config['transform_config']
    transform = transforms.build(transform_config)

    # then build dataset
    dataset_config = config['dataset_config']
    dataset = datasets.build(dataset_config, transform)
    # for debug
    dataset[0]

    sampler = make_data_sampler(dataset, shuffle)
    batch_sampler = make_batch_data_sampler(dataset, sampler, batch_size,
                                            )
    #  collator = BatchCollator()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        #  collate_fn=collator,
        num_workers=num_workers)
    return dataloader
