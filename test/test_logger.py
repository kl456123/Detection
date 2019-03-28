# -*- coding: utf-8 -*-

import sys
sys.path.append('./')
from core.utils.logger import setup_logger




def test_logger():
    logger = setup_logger('test')
    logger.info('asdg')



if __name__=='__main__':
    test_logger()

