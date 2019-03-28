# -*- coding: utf-8 -*-

import json


class Config(object):
    def __init__(self):
        pass

    @staticmethod
    def fromjson(json_file):
        with open(json_file) as f:
            config = json.load(f)
            return config
