# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import model.utils.net_utils as net_utils
from utils.visualize import visualize_bbox
import time


def __change_into_variable(elems, use_gpu=True):
    if use_gpu:
        return [Variable(elem.cuda()) for elem in elems]
    else:
        return [Variable(elem) for elem in elems]


def train(train_config, data_loader, model, optimizer, scheduler, saver):
    start_epoch = train_config['start_epoch']
    max_epochs = train_config['max_epochs']

    disp_interval = train_config['disp_interval']
    clip_gradient = train_config['clip_gradient']

    iters_per_epoch = len(data_loader)

    for epoch in range(start_epoch, max_epochs + 1):
        # setting to train mode
        model.train()
        start = time.time()
        scheduler.step()

        for step, _data in enumerate(data_loader):
            im_data = _data['img']
            im_info = _data['im_info']
            gt_boxes = _data['bbox']
            num_boxes = _data['num']
            # import ipdb
            # ipdb.set_trace()
            # visualize_bbox(im_data.numpy()[0].transpose(1, 2, 0),
            # gt_boxes.numpy()[0, :, :4])

            im_data, im_info, gt_boxes, num_boxes = __change_into_variable(
                [im_data, im_info, gt_boxes, num_boxes])
            # rois, cls_prob, bbox_pred, \
            # rpn_loss_cls, rpn_loss_box, \
            # RCNN_loss_cls, RCNN_loss_bbox, \
            prediction = model(im_data, im_info, gt_boxes, num_boxes)

            # loss
            rpn_loss_cls = prediction['rpn_loss_cls']
            rpn_loss_box = prediction['rpn_loss_bbox']
            RCNN_loss_cls = prediction['RCNN_loss_cls']
            RCNN_loss_bbox = prediction['RCNN_loss_bbox']

            # pred
            rois_label = prediction['rois_label']

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()

            # backward
            optimizer.zero_grad()
            loss.backward()

            net_utils.clip_gradient(model, clip_gradient)
            optimizer.step()

            if step % disp_interval == 0:
                end = time.time()

                loss_rpn_cls = rpn_loss_cls.mean().data[0]
                loss_rpn_box = rpn_loss_box.mean().data[0]
                loss_rcnn_cls = RCNN_loss_cls.mean().data[0]
                loss_rcnn_box = RCNN_loss_bbox.mean().data[0]
                fg_cnt = torch.sum(rois_label.data.ne(0))
                bg_cnt = rois_label.data.numel() - fg_cnt

                print("[epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" % (
                    epoch, step, iters_per_epoch, loss, scheduler.get_lr()[0]))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" %
                      (fg_cnt, bg_cnt, end - start))
                print(
                    "\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f"
                    %
                    (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

                start = time.time()

        checkpoint_name = 'faster_rcnn_{}_{}.pth'.format(epoch, step)
        params_dict = {
            'start_epoch': epoch + 1,
            'model': model,
            'optimizer': optimizer,
        }
        saver.save(params_dict, checkpoint_name)
        end = time.time()
        print(end - start)
