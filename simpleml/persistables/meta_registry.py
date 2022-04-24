'''
Backwards compatible import path, will be removed soon!
'''

__author__ = 'Elisha Yadgaran'


import logging

from simpleml.registries import (
    DATASET_REGISTRY,
    KERAS_REGISTRY,
    METRIC_REGISTRY,
    MODEL_REGISTRY,
    PIPELINE_REGISTRY,
    SIMPLEML_REGISTRY,
    DatasetRegistry,
    KerasRegistry,
    MetaRegistry,
    MetricRegistry,
    ModelRegistry,
    PipelineRegistry,
    Registry,
)

LOGGER = logging.getLogger(__name__)
LOGGER.warning('Importing from a deprecated path! `simpleml.persistables.meta_registry` will be removed in a future release! Update references to `simpleml.registries`')
