'''
Import modules to register class names in global registry

Define convenience classes composed of different mixins
'''

__author__ = 'Elisha Yadgaran'


from .base_pipeline import BasePipeline
from .validation_split_mixins import NoSplitMixin, RandomSplitMixin,\
    ChronologicalSplitMixin, ExplicitSplitMixin


# Mixin implementations for convenience
class BaseNoSplitPipeline(BasePipeline, NoSplitMixin):
    pass


class BaseExplicitSplitPipeline(BasePipeline, ExplicitSplitMixin):
    pass


class BaseRandomSplitPipeline(RandomSplitMixin, BasePipeline):
    # Needs to be used as base class because of MRO initialization
    pass


class BaseChronologicalSplitPipeline(ChronologicalSplitMixin, BasePipeline):
    # Needs to be used as base class because of MRO initialization
    pass
