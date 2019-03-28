# -*- coding: utf-8 -*-

from models.losses import common_loss
import torch

if __name__ == '__main__':
    cls_loss = common_loss.CrossEntropyLoss(reduce=False)

    pred = torch.ones((4, 3))
    weight = torch.ones((4))
    target = torch.ones(4).long()

    targets = {'pred': pred, 'weight': weight, 'target': target}
    print(cls_loss.forward(targets))
    print(cls_loss(targets))
