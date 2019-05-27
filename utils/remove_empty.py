# -*- coding: utf-8 -*-
import os

src = './data/nuscenes_val.txt'
label_dir = '/data/nuscenes_kitti/object/training/label_2/'

with open(src, 'r') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]

lines_set = set(lines)
for line in lines:
    path = os.path.join(label_dir, '{}.txt'.format(line))
    if os.stat(path).st_size == 0:
        lines_set.remove(line)

with open('val.txt', 'w') as f:
    f.write('\n'.join(list(lines_set)))
