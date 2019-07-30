# -*- coding: utf-8 -*-

from PIL import Image
import json
import torch
import numpy as np
import os
from core.utils.logger import setup_logger
import sys
import copy
from utils.parallel_postprocess import mono_3d_postprocess_bbox
from lib.model.roi_layers import nms
from builder import model_builder
import data.transforms.kitti_transform as trans
from utils.box_vis import load_projection_matrix
import argparse
import pprint
from core.saver import Saver


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        default='cfgs/vgg16.yml',
        type=str)
    parser.add_argument(
        '--model', dest='model', help='path to pretrained model', type=str)
    parser.add_argument(
        '--set',
        dest='set_cfgs',
        help='set config keys',
        default=None,
        nargs=argparse.REMAINDER)
    parser.add_argument(
        '--load_dir',
        dest='load_dir',
        help='directory to load models',
        default="/srv/share/jyang375/models",
        type=str)
    parser.add_argument(
        '--cuda', dest='cuda', help='whether use CUDA', action='store_true')
    parser.add_argument(
        '--mGPUs',
        dest='mGPUs',
        help='whether use multiple GPUs',
        action='store_true')
    parser.add_argument(
        '--net', dest='net', help='which base mode to use', type=str)
    parser.add_argument(
        '--parallel_type',
        dest='parallel_type',
        help=
        'which part of model to parallel, 0: all, 1: model before roi pooling',
        default=0,
        type=int)
    parser.add_argument(
        '--checkepoch',
        dest='checkepoch',
        help='checkepoch to load network',
        default=1,
        type=int)
    parser.add_argument(
        '--checkpoint',
        dest='checkpoint',
        help='checkpoint to load network',
        default=10021,
        type=int)
    parser.add_argument(
        '--vis', dest='vis', help='visualization mode', action='store_true')

    parser.add_argument(
        '--img_path',
        dest='img_path',
        help='path to image',
        default='',
        type=str)
    parser.add_argument(
        '--rois_vis',
        dest='rois_vis',
        help='if to visualize rois',
        action='store_true')
    parser.add_argument(
        '--feat_vis',
        dest='feat_vis',
        help='visualize feat or not',
        default=False,
        type=bool)
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='kitti or others',
        type=str,
        default='kitti')
    parser.add_argument(
        '--img_dir',
        dest='img_dir',
        help='directory used for storing imgs',
        type=str)
    parser.add_argument(
        '--calib_file', dest='calib_file', help='calib file', type=str)
    parser.add_argument(
        '--config', dest='config', help='config file(.json)', type=str)
    parser.add_argument(
        "--nms", dest='nms', help='nms to suppress bbox', type=float)
    parser.add_argument(
        "--thresh", dest="thresh", help='thresh for scores', type=float)
    args = parser.parse_args()
    return args


def infer_config_fn(args):
    import glob
    output_dir = args.load_dir + '/' + args.net + '/' + args.dataset
    possible_config = glob.glob(os.path.join(output_dir, '*.json'))
    print(output_dir)
    assert len(possible_config) == 1
    return os.path.join(output_dir, possible_config[0])


class Mono3DInfer(object):
    MEAN_DIMS = {
        'Car': [3.88311640418, 1.62856739989, 1.52563191462],
        'Van': [5.06763659, 1.9007158, 2.20532825],
        'Truck': [10.13586957, 2.58549199, 3.2520595],
        'Pedestrian': [0.84422524, 0.66068622, 1.76255119],
        'Person_sitting': [0.80057803, 0.5983815, 1.27450867],
        'Cyclist': [1.76282397, 0.59706367, 1.73698127],
        'Tram': [16.17150617, 2.53246914, 3.53079012],
        'Misc': [3.64300781, 1.54298177, 1.92320313]
    }

    def __init__(self, args, save=False):
        config = self.generate_config(args)
        self.config = config
        self.model = self.build_model(config)
        self.transforms = self.build_transforms(config['eval_data_config'])

        #  self.logger = setup_logger()
        self.save = save
        self.classes = ['Car']

    def generate_config(self, args):
        if args.config is None:
            # infer it
            args.config = infer_config_fn(args)
        with open(args.config) as f:
            config = json.load(f)

        model_config = config['model_config']
        data_config = config['eval_data_config']
        eval_config = config['eval_config']

        model_config['pretrained'] = False

        assert args.net is not None, 'please select a base model'
        model_config['net'] = args.net

        assert args.load_dir is not None, 'please choose a directory to load checkpoint'
        eval_config['load_dir'] = args.load_dir
        eval_config['feat_vis'] = args.feat_vis

        if args.dataset is not None:
            data_config['name'] = args.dataset

        eval_config['rois_vis'] = args.rois_vis

        if args.nms is not None:
            eval_config['nms'] = args.nms

        if args.thresh is not None:
            eval_config['thresh'] = args.thresh

        if args.img_path:
            dataset_config = data_config['dataset_config']
            # disable dataset file,just use image directly
            dataset_config['dataset_file'] = None
            dataset_config['demo_file'] = args.img_path
            dataset_config['calib_file'] = args.calib_file

        if args.img_dir:
            dataset_config = data_config['dataset_config']
            dataset_config['dataset_file'] = None
            dataset_config['img_dir'] = args.img_dir
            dataset_config['calib_file'] = args.calib_file

        print('Called with args:')
        print(args)

        np.random.seed(eval_config['rng_seed'])

        print('Using config:')
        pprint.pprint({
            'model_config': model_config,
            'data_config': data_config,
            'eval_config': eval_config
        })

        if args.model is not None:
            checkpoint_name = args.model
            input_dir = ''
        else:
            input_dir = eval_config['load_dir'] + "/" + model_config['net'] + "/" + data_config['name']
            if not os.path.exists(input_dir):
                raise Exception(
                    'There is no input directory for loading network from {}'.
                    format(input_dir))
            checkpoint_name = 'faster_rcnn_{}_{}.pth'.format(
                args.checkepoch, args.checkpoint)

        config['checkpoint_name'] = checkpoint_name
        config['input_dir'] = input_dir
        return config

    def build_model(self, config):
        model_config = config['model_config']
        model = model_builder.build(model_config, training=False)
        saver = Saver(config['input_dir'])
        saver.load({'model': model}, config['checkpoint_name'])
        model.eval()
        model.cuda()
        return model

    def build_transforms(self, config):
        trans_cfg = config['transform_config']
        transform = trans.Compose([
            trans.Resize(trans_cfg['crop_size']),
            trans.ToTensor(),
            trans.Normalize(trans_cfg['normal_mean'], trans_cfg['normal_van'])
        ])
        return transform

    def _decompose_project_matrix(self, p2):
        K = p2[:3, :3]
        KT = p2[:, 3]
        T = np.dot(np.linalg.inv(K), KT)
        return K, T

    def _get_mean_dims(self):
        cls_mean_dims = []
        for cls in self.classes:
            cls_mean_dims.append(self.MEAN_DIMS[cls][::-1])
        return np.asarray(cls_mean_dims)

    def preprocess(self, image, p2):
        K, T = self._decompose_project_matrix(p2)
        transform_sample = {
            'img': image,
            'im_scale': 1.0,
            'p2': p2,
            'orig_p2': copy.deepcopy(p2),
            'K': K,
            'T': T
        }
        #  transform_sample.update({'img_orig': np.asarray(image).copy()})

        # get mean dims for encode and decode
        mean_dims = self._get_mean_dims()
        transform_sample['mean_dims'] = torch.from_numpy(mean_dims)
        data = self.transforms(transform_sample)
        im_scale = data['im_scale']
        img = data['img']
        w = img.size()[2]
        h = img.size()[1]
        img_info = torch.FloatTensor([h, w, im_scale])
        data['im_info'] = img_info
        del data['im_scale']
        return data

    def to_batch(self, data, cuda=True):
        for key in data:
            item = data[key][None, ...]
            if isinstance(item, np.ndarray):
                item = torch.from_numpy(item)
            if cuda:
                item = item.cuda()
            data[key] = item
        return data

    def inference(self, image, p2, img_name):
        #  import ipdb
        #  ipdb.set_trace()
        eval_config = self.config['eval_config']
        # prepare data
        data = self.preprocess(image, p2)
        #  data['img'] = data['img'][None]
        data = self.to_batch(data)

        # output
        with torch.no_grad():
            prediction = self.model(data)

        scores = prediction['rcnn_cls_probs']
        rois = prediction['rois_batch']
        bbox_pred = prediction['rcnn_bbox_preds']
        rcnn_3d = prediction['rcnn_3d']

        image_info = data['im_info']
        #  img_file = data['img_name']

        im_scale = image_info[0][2]
        proposals = rois[:, :, 1:]

        box_deltas = bbox_pred

        pred_boxes = self.model.target_assigner.bbox_coder.decode_batch(
            box_deltas.view(eval_config['batch_size'], -1, 4), proposals)

        pred_boxes /= im_scale

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        rois = rois.squeeze()
        rcnn_3d = rcnn_3d.squeeze()

        classes = eval_config['classes']
        thresh = eval_config['thresh']

        dets = []

        for j in range(1, len(classes) + 1):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]

                cls_boxes = pred_boxes[inds, :]
                rcnn_3d_dets = torch.cat(
                    [
                        rcnn_3d[inds, j * 3:j * 3 + 3],
                        rcnn_3d[inds, (len(classes) + 1) * 3:]
                    ],
                    dim=-1)

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)

                # sort
                _, order = torch.sort(cls_scores, 0, True)
                cls_dets = cls_dets[order]
                rcnn_3d_dets = rcnn_3d_dets[order]

                keep = nms(cls_dets[:, :4], cls_dets[:, -1],
                           eval_config['nms'])
                cls_dets = cls_dets[keep.view(-1).long()]
                rcnn_3d_dets = rcnn_3d_dets[keep.view(-1).long()]

                cls_dets = cls_dets.detach().cpu().numpy()

                p2 = data['orig_p2'][0].detach().cpu().numpy()
                rcnn_3d_dets = rcnn_3d_dets.detach().cpu().numpy()

                if eval_config['use_postprocess']:
                    rcnn_3d_dets, location = mono_3d_postprocess_bbox(
                        rcnn_3d_dets,
                        cls_dets,
                        p2,
                        calc_trans=eval_config['calc_trans'])

                dets.append(np.concatenate([cls_dets, rcnn_3d_dets], axis=-1))

            else:
                dets.append([])

        #  import ipdb
        #  ipdb.set_trace()
        if self.save:
            self.save_dets(
                dets,
                img_name,
                'kitti',
                eval_config['eval_out'],
                classes_name=eval_config['classes'])
        return dets

    def save_dets(self,
                  dets,
                  label_info,
                  data_format='kitti',
                  output_dir='',
                  classes_name=['Car']):

        label_info = os.path.basename(label_info)
        label_idx = os.path.splitext(label_info)[0]
        label_file = label_idx + '.txt'
        label_path = os.path.join(output_dir, label_file)
        res_str = []
        kitti_template = '{} -1 -1 -10 {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.8f}'
        with open(label_path, 'w') as f:
            for cls_ind, dets_per_classes in enumerate(dets):
                for det in dets_per_classes:
                    xmin, ymin, xmax, ymax, cf, h, w, l, x, y, z, alpha = det
                    res_str.append(
                        kitti_template.format(classes_name[cls_ind], xmin,
                                              ymin, xmax, ymax, h, w, l, x, y,
                                              z, alpha, cf))
            f.write('\n'.join(res_str))


if __name__ == '__main__':
    # import ipdb
    # ipdb.set_trace()
    image_path = '/data/object/training/image_2/000001.png'
    calib_path = '/data/object/training/calib/000001.txt'
    args = parse_args()
    im = Image.open(image_path)
    infer = Mono3DInfer(args, save=True)
    p2 = load_projection_matrix(calib_path)
    dets = infer.inference(im, p2, image_path)
    #  infer.vis_result(im_cv2, dets, p2)
