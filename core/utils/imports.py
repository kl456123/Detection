# -*- coding: utf-8 -*-
"""
Import helper functions
"""

import os


def get_modules_collection(dirname, exclude=[], include=None):
    """
    This can help import all py files in a module
    """
    import glob
    modules = glob.glob(dirname + '/*.py')
    modules = [
        os.path.basename(f)[:-3] for f in modules
        if os.path.isfile(f) and not f.endswith('__init__.py')
    ]

    modules = list(filter(lambda x: x not in exclude, modules))

    if include is not None:
        modules = list(filter(lambda x: x in include, modules))
    return modules
