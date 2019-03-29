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


def import_file(module_name, file_path):
    import imp
    module = imp.load_source(module_name, file_path)
    return module


def import_dir(*args, **kwargs):
    modules = get_modules_collection(*args, **kwargs)

    for module_name in modules:
        module_path = os.path.join(args[0], module_name + '.py')
        import_file(module_name, module_path)
