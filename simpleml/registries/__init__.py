'''
Import path for the different registries available
'''

__author__ = 'Elisha Yadgaran'


from simpleml.registries.registry import Registry, NamedRegistry
from simpleml.registries.keras_registry import KerasRegistry, KERAS_REGISTRY
from simpleml.registries.sqlalchemy_registry import \
    MetaRegistry, SIMPLEML_REGISTRY, \
    DatasetRegistry, DATASET_REGISTRY, \
    PipelineRegistry, PIPELINE_REGISTRY, \
    ModelRegistry, MODEL_REGISTRY, \
    MetricRegistry, METRIC_REGISTRY


# Registry for save patterns
SAVE_METHOD_REGISTRY = NamedRegistry()
LOAD_METHOD_REGISTRY = NamedRegistry()
