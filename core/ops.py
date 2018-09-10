# -*- coding: utf-8 -*-


def meshgrid(a, b):
    """
    Args:
        a: tensor of shape(N)
        b: tensor of shape(M)
    Returns:
        two tensor of shape(M,N) and shape (M,N)
    """
    a_ = a.repeat(b.numel())
    b_ = b.repeat(a.numel(), 1).t().contiguous().view(-1)
    return a_, b_
