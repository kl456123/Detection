# -*- coding: utf-8 -*-
import time
import torch.nn as nn
import logging
from core.utils import common
import copy


class Trainer(object):
    def __init__(self, train_config, logger=None):
        self.num_iters = train_config['num_iters']
        self.disp_interval = train_config['disp_interval']
        self.checkpoint_interval = train_config['checkpoint_interval']
        self.clip_gradient = train_config['clip_gradient']
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

        self.stats = common.Stats()

    def pre_forward(self, step):
        pass

    def post_forward(self, step):
        pass

    def train(self, data_loader, model, optimizer, scheduler, saver,
              summary_writer):

        self.logger.info('Start training')

        for step, data in enumerate(data_loader):
            if step >= self.num_iters:
                self.logger.info('iteration is done')
                break
            start_time = time.time()
            # to gpu
            data = common.to_cuda(data)

            # forward and backward
            prediction = model(data)
            # loss
            loss_dict = model.loss(prediction, data)

            loss = 0
            for loss_key, loss_val in loss_dict.items():
                loss += loss_val.mean()

            optimizer.zero_grad()
            loss.backward()

            # clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), self.clip_gradient)
            # update weight
            optimizer.step()

            # adjust lr
            scheduler.step()

            self.stats.update_stats(model)

            if step and step % self.disp_interval == 0:
                # display info
                duration_time = time.time() - start_time
                self.logger.info(
                    '[iter {}] time cost: {:.4f}, loss: {:.4f}, lr: {:.2e}'.
                    format(step, duration_time, loss, scheduler.get_lr()[0]))

                # info stats
                self.logger.info(self.stats)
                common.print_loss(loss_dict)
                self.stats.clear_stats()

            if step and step % self.checkpoint_interval == 0:
                # save model
                checkpoint_name = 'detector_{}.pth'.format(step)
                params_dict = {
                    'num_iters': self.num_iters - step,
                    'model': model,
                    'optimizer': optimizer,
                    'scheduler': scheduler
                }
                saver.save(params_dict, checkpoint_name)
