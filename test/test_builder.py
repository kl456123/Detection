# -*- coding: utf-8 -*-

import sys
print(sys.path.append('.'))
from utils import builder


def test_build_transform():
    transform_config = [{'type': 'to_tensor'}]
    transfrom = builder.build_transform(transform_config)


if __name__=='__main__':
    test_build_transform()
