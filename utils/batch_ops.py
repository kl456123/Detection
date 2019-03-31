# -*- coding: utf-8 -*-

import torch
from core import constants


def filter_tensor_container(tensor_container, mask):
    if isinstance(tensor_container, dict):
        for key, tensor in tensor_container.items():
            tensor_container[key] = filter_tensor_container(tensor, mask)
        return tensor_container
    elif isinstance(tensor_container, list):
        for idx, item in enumerate(tensor_container):
            tensor_container[idx] = filter_tensor_container(item, mask)
        return tensor_container
    elif isinstance(tensor_container, torch.Tensor):
        return filter_tensor(tensor_container, mask)
    else:
        raise ValueError('cannot filter tensor container')


def filter_tensor(tensor, mask):
    """
    Note that each row in mask has the same number of pos items
    so that masked_tensor can be reshaped
    Args:
        tensor: shape(N,M) or shape(N,M,K)
        mask: shape(N,M)
    Returns:
        masked_tensor: shape(N, M') or shape(N,M',K)
    """
    masked_tensor = tensor[mask]
    batch_size = tensor.shape[0]
    if len(tensor.shape) == 3:
        last_dim = tensor.shape[-1]
        masked_tensor_shape = (batch_size, -1, last_dim)
    else:
        masked_tensor_shape = (batch_size, -1)
    return masked_tensor.view(masked_tensor_shape)


def make_match_dict(primary, non_prime=[]):
    proposals_dict = {}
    proposals_dict[constants.KEY_PRIMARY] = primary
    proposals_dict[constants.KEY_NON_PRIME] = non_prime
    return proposals_dict
