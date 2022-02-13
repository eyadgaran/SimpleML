'''
Pipeline Library support for Scikit-Learn
'''

__author__ = 'Elisha Yadgaran'


from .base import SklearnPipeline
from .split_pipelines import (ExplicitSplitSklearnPipeline,
                              RandomSplitSklearnPipeline)
