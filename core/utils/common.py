# -*- coding: utf-8 -*-


def build(config, registry, *args, **kwargs):
    if 'type' not in config:
        raise ValueError('config has no type, it can not be builded')
    class_type = config['type']
    if class_type not in registry:
        raise TypeError(
            "unknown {} type {}!".format(registry.name, class_type))
    registered_class = registry[class_type]
    # use config to build it
    return registered_class(config, *args, **kwargs)
