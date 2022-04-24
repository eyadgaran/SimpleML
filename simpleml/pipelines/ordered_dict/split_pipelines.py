'''
Derivative OrderedDict pipeline wrappers
'''

__author__ = 'Elisha Yadgaran'


from typing import Any, List

from simpleml.pipelines.ordered_dict.base import OrderedDictPipeline
from simpleml.pipelines.validation_split_mixins import (
    ExplicitSplitMixin,
    RandomSplitMixin,
)


class RandomSplitOrderedDictPipeline(RandomSplitMixin, OrderedDictPipeline):
    pass


class ExplicitSplitOrderedDictPipeline(ExplicitSplitMixin, OrderedDictPipeline):
    pass
