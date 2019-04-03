# -*- coding: utf-8 -*-

import json
import os


class Config(object):
    def __init__(self):
        pass

    @staticmethod
    def fromjson(json_file):
        with open(json_file) as f:
            config = json.load(f)
            return config

    @staticmethod
    def infer_fromdir(config_dir):
        import glob
        possible_config = glob.glob(os.path.join(config_dir, '*.json'))
        num_possible_configs = len(possible_config)
        if num_possible_configs > 1:
            raise ValueError(
                'too many possible configs in {}'.format(config_dir))
        if num_possible_configs == 0:
            raise ValueError(
                'no any possible config can be used in {}'.format(config_dir))
        return os.path.join(config_dir, possible_config[0])
