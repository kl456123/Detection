# -*- coding: utf-8 -*-

import os
import argparse

img_dir = '/data/object/training/image_2'
# img_dir = '/data/liangxiong/COCO2017/val2017'
kitti_dir = 'results/data/'
anchors_dir = 'results/anchors/'
rois_dir = 'results/rois/'
lbl_dir = '/data/object/training/label_2'


def vis_all(kitti_dir, title='test', dis_lbl=False, resume=None):
    for kitti in sorted(os.listdir(kitti_dir)):
        sample_idx = os.path.splitext(kitti)[0]
        if resume is not None and (not int(sample_idx) == resume):
            continue
        else:
            resume = None
        print('current image idx: {}'.format(sample_idx))
        img_path = os.path.join(img_dir, '{}.png'.format(sample_idx))
        kitti_path = os.path.join(kitti_dir, '{}.txt'.format(sample_idx))
        if dis_lbl:
            lbl_path = os.path.join(lbl_dir, '{}.txt'.format(sample_idx))
            command = 'python utils/visualize.py --img {} --kitti {} --title {} --label {}'.format(
                img_path, kitti_path, title, lbl_path)
        else:
            command = 'python utils/visualize.py --img {} --kitti {} --title {}'.format(
                img_path, kitti_path, title)
        os.system(command)
        input("Press Enter to continue...")


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--type', dest='label_type', type=str, default='pred')
    argparser.add_argument(
        '--display_label', dest='dis_lbl', type=bool, default=False)
    argparser.add_argument('--resume', dest='resume', type=int)

    args = argparser.parse_args()
    if args.label_type == 'pred':
        label_dir = kitti_dir
        title = 'pred'
    elif args.label_type == 'anchor':
        label_dir = anchors_dir
        title = 'anchor'
    elif args.label_type == 'roi':
        label_dir = rois_dir
        title = 'roi'
    else:
        raise ValueError('unknown label type!')

    resume = args.resume

    vis_all(label_dir, title, args.dis_lbl, resume=resume)


if __name__ == '__main__':
    main()
