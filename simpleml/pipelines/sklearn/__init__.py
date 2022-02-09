'''
Pipeline Library support for Scikit-Learn
'''

__author__ = 'Elisha Yadgaran'


from .base import SklearnExternalPipeline
from .split_pipelines import (ExplicitSplitSklearnPipeline,
                              RandomSplitSklearnPipeline)
