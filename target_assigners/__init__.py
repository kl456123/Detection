# -*- coding: utf-8 -*-

from core.utils.imports import get_modules_collection
import os

include = ['target_assigner']

__all__ = get_modules_collection(os.path.dirname(__file__), include=include)
