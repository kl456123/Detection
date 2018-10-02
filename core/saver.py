# -*- coding: utf-8 -*-

import os
import torch
import shutil


class Saver():
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def get_checkpoint_path(self, checkpoint_name):
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        return checkpoint_path

    def load(self, params_dict, checkpoint_name):
        checkpoint_path = self.get_checkpoint_path(checkpoint_name)
        print(("loading checkpoint %s" % (checkpoint_path)))

        checkpoint = torch.load(checkpoint_path)
        for name, module in list(params_dict.items()):
            if name in checkpoint:
                if hasattr(module, 'load_state_dict'):
                    module_dict = module.state_dict()
                    checkpoint_dict = {
                        k: v
                        for k, v in checkpoint[name].items()
                        if k in module_dict
                    }
                    module_dict.update(checkpoint_dict)
                    module.load_state_dict(module_dict)
                    # module.load_state_dict(checkpoint[name])
                else:
                    params_dict[name] = checkpoint[name]
            else:
                print(('module:{} can not be loaded'.format(name)))

        print(("loaded checkpoint %s" % (checkpoint_name)))

    def save(self, params_dict, checkpoint_name, is_best=False):
        checkpoint_path = self.get_checkpoint_path(checkpoint_name)
        state = {}
        for name, module in list(params_dict.items()):
            if isinstance(module, torch.nn.parallel.DataParallel):
                state[name] = module.module.state_dict()
            elif hasattr(module, 'state_dict'):
                state[name] = module.state_dict()
            else:
                state[name] = module
        self.save_checkpoint(state, is_best, checkpoint_path)
        print(('save model: {}'.format(checkpoint_name)))

    @staticmethod
    def save_checkpoint(state, is_best=False, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')
