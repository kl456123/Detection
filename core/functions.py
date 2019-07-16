# -*- coding: utf-8 -*-

import torch
from torch.autograd import Function


class SparseToDense(Function):
    @staticmethod
    def forward(ctx, index, net_input, shape):
        index, net_input = index.detach(), net_input.detach()

        target_tensor = torch.zeros(shape)
        target_tensor[index] = net_input

        ctx.save_for_backward(index)
        return target_tensor

    @staticmethod
    def backward(ctx, grad_output):
        index = ctx.saved_tensors
        return grad_output[index]


if __name__ == '__main__':
    pass
