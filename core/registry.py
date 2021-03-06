# -*- coding: utf-8 -*-


class Registry(dict):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        super().__init__(*args, **kwargs)

    def register(self, module_name, module=None):
        # used as fn
        if module is not None:
            Registry._register_generic(self, module_name, module)
            return

        # used as decorator
        def register_fn(fn):
            Registry._register_generic(self, module_name, fn)
            return fn

        return register_fn

    @staticmethod
    def _register_generic(module_dict, module_name, module):
        if module_name in module_dict:
            if type(module) is not type(module_dict[module_name]):
                raise RuntimeError('repeat register the same name module')
        module_dict[module_name] = module
