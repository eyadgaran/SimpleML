'''
Import path for the different registries available
'''

__author__ = 'Elisha Yadgaran'


from simpleml.registries.keras_registry import KERAS_REGISTRY, KerasRegistry
from simpleml.registries.registry import NamedRegistry, Registry
from simpleml.registries.sqlalchemy_registry import (
    DATASET_REGISTRY,
    METRIC_REGISTRY,
    MODEL_REGISTRY,
    PIPELINE_REGISTRY,
    SIMPLEML_REGISTRY,
    DatasetRegistry,
    MetaRegistry,
    MetricRegistry,
    ModelRegistry,
    PipelineRegistry,
)

# Registry for save patterns
SAVE_METHOD_REGISTRY = NamedRegistry()
LOAD_METHOD_REGISTRY = NamedRegistry()

# Registry for dynamic system filepaths
FILEPATH_REGISTRY = NamedRegistry()
