# -*- coding: utf-8 -*-

import logging

class Logger(logging.Logger):
    def __init__(self, logger_config):
        super().__init__()
        self.logger_name = logger_config['name']
        # both logger and handler use the same level
        self.logger_level = logger_config['level']
        self.save_dir = logger_config.get('save_dir')
        self.filename = logger_config.get('log.txt')

        # stream handler
        ch = logging.StreamHandler(stream=sys.stdout)
        self.register_handler(ch)

        if self.save_dir:
            fh = logging.FileHandler(os.path.join(save_dir, self.filename))
            self.register_handler(fh)

    def register_handler(self, handler):
        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        handler.setLevel(self.logger_level)
        handler.setFormatter(formatter)
        self.addHandler(handler)

    def init_logger(self):
        self.setLevel(self.logger_level)


