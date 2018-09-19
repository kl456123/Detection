# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
from utils.visualize import visualize_bbox
import time
import torch.nn as nn


def to_cuda(target):
    if isinstance(target, list):
        return [to_cuda(e) for e in target]
    elif isinstance(target, dict):
        return {key: to_cuda(target[key]) for key in target}
    elif isinstance(target, torch.Tensor):
        return target.cuda()


def train(train_config, data_loader, model, optimizer, scheduler, saver,
          summary_writer):
    start_epoch = train_config['start_epoch']
    max_epochs = train_config['max_epochs']

    disp_interval = train_config['disp_interval']
    clip_gradient = train_config['clip_gradient']

    iters_per_epoch = len(data_loader)
    total_step = 0

    for epoch in range(start_epoch, max_epochs + 1):
        # setting to train mode
        start = time.time()
        scheduler.step()

        matched = 0
        num_gt = 0

        for step, data in enumerate(data_loader):

            data = to_cuda(data)

            prediction = model(data)
            # proposals_batch = prediction['proposals_batch'][0]
            # rois = prediction['rois_batch'][0]
            # proposals_batch = rois.data[:, 1:5]
            # anchors = prediction['anchors'][0]
            # img = data['img'].permute(0, 2, 3, 1)
            # import ipdb
            # ipdb.set_trace()
            # visualize_bbox(
            # img.cpu().numpy()[0], proposals_batch.cpu().numpy(), save=True)

            # loss
            loss_dict = model.loss(prediction, data)
            loss_dict.update(prediction)
            rpn_cls_loss = loss_dict['rpn_cls_loss']
            rpn_bbox_loss = loss_dict['rpn_bbox_loss']
            rcnn_cls_loss = loss_dict['rcnn_cls_loss']
            rcnn_bbox_loss = loss_dict['rcnn_bbox_loss']

            # pred
            rois_label = loss_dict['rcnn_cls_targets']

            loss = rpn_cls_loss.mean() + rpn_bbox_loss.mean() \
                + rcnn_cls_loss.mean() + rcnn_bbox_loss.mean()

            # backward
            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), clip_gradient)
            optimizer.step()

            stat = model.target_assigner.stat
            matched += stat['matched']
            num_gt += stat['num_gt']
            if step % disp_interval == 0:
                end = time.time()

                # rpn_ap = prediction['rpn_ap']
                # rpn_ar = prediction['rpn_ar']
                # rpn_ap = rpn_ap.mean().data[0]
                # rpn_ar = rpn_ar.mean().data[0]

                # rcnn_ap = prediction['rcnn_ap']
                # rcnn_ar = prediction['rcnn_ar']
                # rcnn_ap = rcnn_ap.mean().data[0]
                # rcnn_ar = rcnn_ar.mean().data[0]

                rpn_cls_loss = rpn_cls_loss.mean().item()
                rpn_bbox_loss = rpn_bbox_loss.mean().item()
                rcnn_cls_loss = rcnn_cls_loss.mean().item()
                rcnn_bbox_loss = rcnn_bbox_loss.mean().item()
                total_loss = loss.item()

                # # summary loss
                summary_writer.add_scalar('loss/rpn_cls_loss', rpn_cls_loss,
                                          total_step + step)
                summary_writer.add_scalar('loss/rpn_bbox_loss', rpn_bbox_loss,
                                          total_step + step)
                summary_writer.add_scalar('loss/rcnn_cls_loss', rcnn_cls_loss,
                                          total_step + step)
                summary_writer.add_scalar('loss/rcnn_bbox_loss',
                                          rcnn_bbox_loss, total_step + step)
                summary_writer.add_scalar('loss/total_loss', total_loss,
                                          total_step + step)

                # # summary metric
                # summary_writer.add_scalar('metric/rpn_ap', rpn_ap)
                summary_writer.add_scalar('metric/rpn_ar', matched / num_gt,
                                          total_step + step)
                # summary_writer.add_scalar('metric/rcnn_ap', rcnn_ap)
                # summary_writer.add_scalar('metric/rcnn_ar', rcnn_ar)

                # summary histogram of predict
                # rpn_cls_probs = prediction['rpn_cls_probs']
                # summary_writer.add_histogram('predict/rpn_cls_probs',
                # rpn_cls_probs)
                # rpn_bbox_preds = prediction['rpn_bbox_preds']
                # summary_writer.add_histogram('predict/rpn_bbox_preds',
                # rpn_bbox_preds)
                # rcnn_cls_probs = prediction['rcnn_cls_probs']
                # summary_writer.add_histogram('predict/rcnn_cls_probs',
                # rcnn_cls_probs)
                # rcnn_bbox_preds = prediction['rcnn_bbox_preds']
                # summary_writer.add_histogram('predict/rcnn_bbox_preds',
                # rcnn_bbox_preds)

                # may be float point number
                fg_cnt = torch.sum(rois_label > 0)
                bg_cnt = rois_label.numel() - fg_cnt

                print(("[epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" %
                       (epoch, step, iters_per_epoch, loss,
                        scheduler.get_lr()[0])))
                print(("\t\t\tfg/bg=(%d/%d), time cost: %f" %
                       (fg_cnt, bg_cnt, end - start)))
                print((
                    "\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f"
                    % (rpn_cls_loss, rpn_bbox_loss, rcnn_cls_loss,
                       rcnn_bbox_loss)))
                print(("\t\t\tmatched_gt/all_gt/average recall({}/{}/{:.4f}): "
                       ).format(matched, num_gt, matched / num_gt))
                # reset
                matched = 0
                num_gt = 0

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
