'''
Module to query registry and retrieve persistables from wherever they are
stored.
'''

from simpleml.datasets.raw_datasets.base_raw_dataset import BaseRawDataset
from simpleml.pipelines.dataset_pipelines.base_dataset_pipeline import BaseDatasetPipeline
from simpleml.datasets.processed_datasets.base_processed_dataset import BaseProcessedDataset
from simpleml.pipelines.production_pipelines.base_production_pipeline import BaseProductionPipeline
from simpleml.models.base_model import BaseModel
from simpleml.metrics.base_metric import BaseMetric
from simpleml.utils.errors import SimpleMLError


__author__ = 'Elisha Yadgaran'


class PersistableLoader(object):
    '''
    Wrapper class to load various persistables

    Sqlalchemy-mixins active record style allows for keyword based filtering:
        `BaseClass.where(**filters).order_by(**ordering).first()`
    '''
    @staticmethod
    def load_persistable(cls, filters):
        persistable = cls.where(**filters).order_by(cls.version.desc()).first()
        if persistable is not None:
            persistable.load(load_externals=False)
            return persistable
        else:
            raise SimpleMLError('No persistable found for specified filters: {}'.format(filters))

    @classmethod
    def load_raw_dataset(cls, name='default', **filters):
        filters['name'] = name
        return cls.load_persistable(BaseRawDataset, filters)

    @classmethod
    def load_dataset_pipeline(cls, name='default', **filters):
        filters['name'] = name
        return cls.load_persistable(BaseDatasetPipeline, filters)

    @classmethod
    def load_dataset(cls, name='default', **filters):
        filters['name'] = name
        return cls.load_persistable(BaseProcessedDataset, filters)

    @classmethod
    def load_pipeline(cls, name='default', **filters):
        filters['name'] = name
        return cls.load_persistable(BaseProductionPipeline, filters)

    @classmethod
    def load_model(cls, name='default', **filters):
        filters['name'] = name
        return cls.load_persistable(BaseModel, filters)

    @classmethod
    def load_metric(cls, name, model_id, **filters):
        filters['name'] = name
        filters['model_id'] = model_id
        return cls.load_persistable(BaseMetric, filters)
