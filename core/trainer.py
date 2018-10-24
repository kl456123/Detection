# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
from utils.visualize import visualize_bbox
import time
import torch.nn as nn
import sys


def to_cuda(target):
    if isinstance(target, list):
        return [to_cuda(e) for e in target]
    elif isinstance(target, dict):
        return {key: to_cuda(target[key]) for key in target}
    elif isinstance(target, torch.Tensor):
        return target.cuda()


def print_loss(loss_dict):
    print_num = 0
    for key, val in loss_dict.items():
        if print_num % 3 == 0:
            sys.stdout.write("\t\t\t")
        sys.stdout.write("{}: {:.4f}\t".format(key, val.mean().item()))
        print_num += 1
        if print_num % 3 == 0:
            sys.stdout.write("\n")
    if print_num % 3:
        sys.stdout.write("\n")


def train(train_config, data_loader, model, optimizer, scheduler, saver,
          summary_writer):
    start_epoch = train_config['start_epoch']
    max_epochs = train_config['max_epochs']

    disp_interval = train_config['disp_interval']
    clip_gradient = train_config['clip_gradient']

    iters_per_epoch = len(data_loader)

    for epoch in range(start_epoch, max_epochs + 1):
        total_step = (epoch - 1) * len(data_loader)
        # setting to train mode
        start = time.time()
        scheduler.step()

        matched = 0
        num_gt = 0
        num_det = 0
        num_tp = 0
        matched_thresh = 0

        for step, data in enumerate(data_loader):

            data = to_cuda(data)

            prediction = model(data)
            # proposals_batch = prediction['proposals_batch'][0]
            # rois = prediction['rois_batch'][0]
            # proposals_batch = rois.data[:, 1:5]
            # print('num of bbox: {}'.format(proposals_batch.shape[0]))
            # anchors = prediction['anchors']
            # img = data['img'].permute(0, 2, 3, 1)

            # visualize_bbox(
            # img.cpu().numpy()[0], anchors.cpu().numpy(), save=True)

            # loss
            loss_dict = model.loss(prediction, data)

            loss = 0
            for loss_key, loss_val in loss_dict.items():
                loss += loss_val.mean()

            # pred
            rois_label = prediction['rcnn_reg_weights']

            # backward
            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), clip_gradient)
            optimizer.step()

            # statistics
            stat = model.target_assigner.stat
            matched += stat['matched']
            num_gt += stat['num_gt']
            num_det += stat['num_det']
            num_tp += stat['num_tp']
            matched_thresh += stat['matched_thresh']
            if step % disp_interval == 0:
                end = time.time()

                # # summary loss
                loss_dict['total_loss'] = loss
                summary_writer.add_scalar_dict(loss_dict, total_step + step)

                # # summary metric
                # summary_writer.add_scalar('metric/rpn_ap', rpn_ap)
                summary_writer.add_scalar('metric/rpn_ar', matched / num_gt,
                                          total_step + step)
                if num_det == 0:
                    precision = 0
                else:
                    precision = num_tp / num_det
                summary_writer.add_scalar('metric/rcnn_ap', precision,
                                          total_step + step)
                summary_writer.add_scalar('metric/rcnn_ar', matched_thresh /
                                          num_gt, total_step + step)

                # may be float point number
                fg_cnt = torch.sum(rois_label > 0)
                bg_cnt = rois_label.numel() - fg_cnt

                print(("[epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" %
                       (epoch, step, iters_per_epoch, loss,
                        scheduler.get_lr()[0])))
                print(("\t\t\tfg/bg=(%d/%d), time cost: %f" %
                       (fg_cnt, bg_cnt, end - start)))
                print_loss(loss_dict)

                print(("\t\t\tmatched_gt/all_gt/average recall({}/{}/{:.4f}): "
                       ).format(matched, num_gt, matched / num_gt))
                print(("\t\t\tnum_tp/num_det/average precision({}/{}/{:.4f}): "
                       ).format(num_tp, num_det, precision))
                print((
                    "\t\t\tmatched_gt_thresh/all_gt/average recall_thresh({}/{}/{:.4f}): "
                ).format(matched_thresh, num_gt, matched_thresh / num_gt))
                # reset
                matched = 0
                num_gt = 0
                num_tp = 0
                num_det = 0
                matched_thresh = 0

                start = time.time()

        checkpoint_name = 'faster_rcnn_{}_{}.pth'.format(epoch, step)
        params_dict = {
            'start_epoch': epoch + 1,
            'model': model,
            'optimizer': optimizer,
        }
        saver.save(params_dict, checkpoint_name)
        end = time.time()
        total_step += step  # epoch level
