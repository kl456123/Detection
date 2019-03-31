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


def change_lr(lr, optimizer):
    for param_group in optimizer.param_groups:
        # used for scheduler
        param_group['initial_lr'] = lr


def to_cuda(target):
    if isinstance(target, list):
        return [to_cuda(e) for e in target]
    elif isinstance(target, dict):
        return {key: to_cuda(target[key]) for key in target}
    elif isinstance(target, torch.Tensor):
        return target.cuda()
    else:
        # dont change
        return target


def print_loss(loss_dict):
    print_num = 0
    for key, val in loss_dict.items():
        if print_num % 3 == 0:
            sys.stdout.write("\t\t\t")
        sys.stdout.write("{}: {:.4f}\t".format(key, val.mean().item()))
        print_num += 1
        if print_num % 3 == 0:
            sys.stdout.write("\n")
    if print_num % 3:
        sys.stdout.write("\n")
