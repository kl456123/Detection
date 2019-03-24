# -*- coding: utf-8 -*-
"""
Import helper functions
"""

import os


def get_modules_collection(dirname):
    """
    This can help import all py files in a module
    """
    import glob
    modules = glob.glob(dirname + '/*.py')
    return [os.path.basename(f)[:-3] for f in modules if os.path.isfile(f) and not f.endswith('__init__.py')]
