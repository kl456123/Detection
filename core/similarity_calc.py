# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod


class SimilarityCalc(object):
    def __init__(self, config):
        pass

    @abstractmethod
    def compare_batch(self):
        pass
