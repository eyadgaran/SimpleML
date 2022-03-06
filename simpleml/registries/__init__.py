"""
Import path for the different registries available
"""

__author__ = "Elisha Yadgaran"


from simpleml.registries.keras_registry import KERAS_REGISTRY, KerasRegistry
from simpleml.registries.persistable_registry import (DATASET_REGISTRY,
                                                      METRIC_REGISTRY,
                                                      MODEL_REGISTRY,
                                                      PIPELINE_REGISTRY,
                                                      SIMPLEML_REGISTRY,
                                                      DatasetRegistry,
                                                      MetricRegistry,
                                                      ModelRegistry,
                                                      PersistableRegistry,
                                                      PipelineRegistry)
from simpleml.registries.registry import NamedRegistry, Registry

# Registry for save patterns
SAVE_METHOD_REGISTRY = NamedRegistry()
LOAD_METHOD_REGISTRY = NamedRegistry()

# Registry for dynamic system filepaths
FILEPATH_REGISTRY = NamedRegistry()

# Registry for ORM Classes
ORM_REGISTRY = NamedRegistry()
