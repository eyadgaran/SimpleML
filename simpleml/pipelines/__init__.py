'''
Import modules to register class names in global registry

Define convenience classes composed of different mixins
'''

__author__ = 'Elisha Yadgaran'


from .base_pipeline import Pipeline, GeneratorPipeline
from .validation_split_mixins import NoSplitMixin, RandomSplitMixin,\
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


# Generator implementations
class NoSplitGeneratorPipeline(GeneratorPipeline, NoSplitMixin):
    pass


class ExplicitSplitGeneratorPipeline(GeneratorPipeline, ExplicitSplitMixin):
    pass


class RandomSplitGeneratorPipeline(RandomSplitMixin, GeneratorPipeline):
    # Needs to be used as base class because of MRO initialization
    pass


class ChronologicalSplitGeneratorPipeline(ChronologicalSplitMixin, GeneratorPipeline):
    # Needs to be used as base class because of MRO initialization
    pass
