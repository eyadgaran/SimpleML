"""
ORM package

Focuses on all database related interaction, intentionally separated to allow
parallel Persistable objects to only deal with the glue interactions across
python types and libraries

Each mapped Persistable table model has a 1:1 parallel class
defined as a native python object
"""

__author__ = 'Elisha Yadgaran'

# explicitly register orm - can be overwritten for alternate ORM implementations
from simpleml.registries import ORM_REGISTRY

from .dataset import ORMDataset
from .initialization import (AlembicDatabase, BaseDatabase,
                             BinaryStorageDatabase, Database, DatasetDatabase)
from .metric import ORMMetric
from .model import ORMModel
from .pipeline import ORMPipeline

ORM_REGISTRY.register('DATASET', ORMDataset)
ORM_REGISTRY.register('PIPELINE', ORMPipeline)
ORM_REGISTRY.register('MODEL', ORMModel)
ORM_REGISTRY.register('METRIC', ORMMetric)
