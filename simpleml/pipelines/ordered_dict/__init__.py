'''
Pipeline Library support for native python dictionaries
'''

__author__ = 'Elisha Yadgaran'


from .base import OrderedDictPipeline
from .split_pipelines import (
    ExplicitSplitOrderedDictPipeline,
    RandomSplitOrderedDictPipeline,
)
