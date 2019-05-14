# -*- coding: utf-8 -*-
import time
import torch.nn as nn
import logging
from core.utils import common
import copy
from core import constants


class Trainer(object):
    def __init__(self, train_config, logger=None):
        self.num_iters = train_config['num_iters']
        self.disp_interval = train_config['disp_interval']
        self.checkpoint_interval = train_config['checkpoint_interval']
        self.clip_gradient = train_config['clip_gradient']
        self.start_iters = train_config['start_iters']
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

        batch_size = data_loader.batch_sampler.batch_size
        self.logger.info('Start training')
        self.logger.info('batch size: {}'.format(batch_size))
        # check batch_size, disp_interval and checkpoint_interval
        assert self.checkpoint_interval % batch_size == 0, \
            'checkpoint_interval({}) cannot be mod by batch_size({})'.format(
                self.checkpoint_interval, batch_size)
        assert self.disp_interval % batch_size == 0, \
            'disp_interval({}) cannot be mod by batch_size({})'.format(
                self.disp_interval, batch_size)
        # start from 1
        start_iters = max(1, self.start_iters // batch_size)
        for step, data in enumerate(data_loader, start_iters):
            # truly step
            step = step * batch_size
            if step > self.num_iters:
                self.logger.info('iteration is done')
                break
            start_time = time.time()
            # to gpu
            data = common.to_cuda(data)

            # forward and backward
            prediction, loss_dict = model(data)
            # loss
            #  loss_dict = model.loss(prediction, data)

            loss = 0
            for loss_key, loss_val in loss_dict.items():
                loss += loss_val.mean()
                # update loss dict
                loss_dict[loss_key] = loss_val.mean()

            optimizer.zero_grad()
            loss.backward()

            # clip gradients
            nn.utils.clip_grad_norm_(model.parameters(), self.clip_gradient)
            # update weight
            optimizer.step()

            # adjust lr
            # step by iters
            scheduler.step(step)

            self.stats.update_stats(prediction[constants.KEY_STATS])

            if step % self.disp_interval == 0:
                # display info
                duration_time = time.time() - start_time
                self.logger.info(
                    '[iter {}] time cost: {:.4f}, loss: {:.4f}, lr: {:.2e}'.
                    format(step, duration_time, loss, scheduler.get_lr()[0]))

                # info stats
                self.logger.info(self.stats)
                self.logger.info(common.loss_dict_to_str(loss_dict))

                # summary writer
                # loss
                loss_dict.update({'total_loss': loss})
                summary_writer.add_scalar_dict(loss_dict, step)

                # metric
                summary_writer.add_scalar_dict(self.stats.get_summary_dict(),
                                               step)
                self.stats.clear_stats()

            if step % self.checkpoint_interval == 0:
                # save model
                checkpoint_name = 'detector_{}.pth'.format(step)
                params_dict = {
                    'start_iters': step + batch_size,
                    'model': model,
                    'optimizer': optimizer,
                    'scheduler': scheduler
                }
                saver.save(params_dict, checkpoint_name)
                self.logger.info('checkpoint {} saved'.format(checkpoint_name))
