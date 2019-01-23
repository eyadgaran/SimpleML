'''
Module to query registry and retrieve persistables from wherever they are
stored.
'''

from simpleml.datasets.base_dataset import Dataset
from simpleml.pipelines.base_pipeline import Pipeline
from simpleml.models.base_model import Model
from simpleml.metrics.base_metric import Metric
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
    def load_dataset(cls, name='default', **filters):
        filters['name'] = name
        return cls.load_persistable(Dataset, filters)

    @classmethod
    def load_pipeline(cls, name='default', **filters):
        filters['name'] = name
        return cls.load_persistable(Pipeline, filters)

    @classmethod
    def load_model(cls, name='default', **filters):
        filters['name'] = name
        return cls.load_persistable(Model, filters)

    @classmethod
    def load_metric(cls, name, model_id, **filters):
        filters['name'] = name
        filters['model_id'] = model_id
        return cls.load_persistable(Metric, filters)
