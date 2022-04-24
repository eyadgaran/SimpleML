'''
Derivative Sklearn pipeline wrappers
'''

__author__ = 'Elisha Yadgaran'


from typing import Any, List

from simpleml.pipelines.sklearn.base import SklearnPipeline
from simpleml.pipelines.validation_split_mixins import (
    ExplicitSplitMixin,
    RandomSplitMixin,
)


class RandomSplitSklearnPipeline(RandomSplitMixin, SklearnPipeline):
    '''
    Pipeline Wrapper with support for projected random splits on dataset
    Useful to create a train/test/validation split on any dataset
    '''
    pass


class ExplicitSplitSklearnPipeline(ExplicitSplitMixin, SklearnPipeline):
    pass
