# -*- coding: utf-8 -*-
import os

src_txt = './data/nuscenes_train.txt'
bad_txt = './bad.txt'
num = len('7a6b6e8af79d45c78614e2e6acf14c89')

with open(bad_txt, 'r') as f:
    bad_lines = f.readlines()
    bad_lines = [os.path.splitext(line.strip())[0][-num:] for line in bad_lines]

with open(src_txt, 'r') as f:
    src_lines = f.readlines()
    src_lines = [line.strip() for line in src_lines]

src_set = set(src_lines)

for bad_line in bad_lines:
    if bad_line in src_set:
        src_set.remove(bad_line)

src_lines = list(src_set)

with open('fine.txt', 'w') as f:
    f.write('\n'.join(src_lines))
