'''
Backwards compatible import path, will be removed soon!
'''

__author__ = 'Elisha Yadgaran'


import logging
from simpleml.registries import Registry, \
    KerasRegistry, KERAS_REGISTRY, \
    MetaRegistry, SIMPLEML_REGISTRY, \
    DatasetRegistry, DATASET_REGISTRY, \
    PipelineRegistry, PIPELINE_REGISTRY, \
    ModelRegistry, MODEL_REGISTRY, \
    MetricRegistry, METRIC_REGISTRY


LOGGER = logging.getLogger(__name__)
LOGGER.warning('Importing from a deprecated path! `simpleml.persistables.meta_registry` will be removed in a future release! Update references to `simpleml.registries`')
