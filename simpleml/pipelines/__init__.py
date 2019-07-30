'''
Import modules to register class names in global registry

Define convenience classes composed of different mixins
'''

__author__ = 'Elisha Yadgaran'


from .base_pipeline import Pipeline, AbstractPipeline, DatasetSequence, TransformedSequence
from .validation_split_mixins import Split, SplitContainer, NoSplitMixin, RandomSplitMixin,\
    ChronologicalSplitMixin, ExplicitSplitMixin


# Mixin implementations for convenience
class NoSplitPipeline(Pipeline, NoSplitMixin):
    pass


class ExplicitSplitPipeline(Pipeline, ExplicitSplitMixin):
    pass


class RandomSplitPipeline(RandomSplitMixin, Pipeline):
    # Needs to be used as base class because of MRO initialization
    pass


class ChronologicalSplitPipeline(ChronologicalSplitMixin, Pipeline):
    # Needs to be used as base class because of MRO initialization
    pass
