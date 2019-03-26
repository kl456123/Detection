# -*- coding: utf-8 -*-

from utils.registry import SCHEDULERS


@SCHEDULERS.register('step')
def build_step_scheduler(scheduler_config):
    pass
