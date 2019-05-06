'''
Module to query registry and retrieve persistables from wherever they are
stored.
'''

__author__ = 'Elisha Yadgaran'


from simpleml.datasets.base_dataset import Dataset
from simpleml.pipelines.base_pipeline import Pipeline
from simpleml.models.base_model import Model
from simpleml.metrics.base_metric import Metric
from simpleml.utils.errors import SimpleMLError
from simpleml.utils.library_versions import INSTALLED_LIBRARIES

import logging

LOGGER = logging.getLogger(__name__)


class PersistableLoader(object):
    '''
    Wrapper class to load various persistables

    Sqlalchemy-mixins active record style allows for keyword based filtering:
        `BaseClass.where(**filters).order_by(**ordering).first()`
    '''
    @classmethod
    def load_persistable(cls, persistable_class, filters):
        persistable = persistable_class.where(**filters).order_by(persistable_class.version.desc()).first()
        if persistable is not None:
            cls.validate_environment(persistable)
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

    @staticmethod
    def validate_environment(persistable):
        training_env = persistable.library_versions
        scoring_env = INSTALLED_LIBRARIES

        mismatches = []

        for package, version in training_env.items():
            if package not in scoring_env:
                mismatches.append({'package': package, 'expected_version': version, 'version': 'None'})
            else:
                if version != scoring_env[package]:
                    mismatches.append({'package': package, 'expected_version': version, 'version': scoring_env[package]})

        warning_msg = 'Attempted to score with different dependencies than training, proceed at your own risk'

        if mismatches:
            LOGGER.warning(warning_msg)
            for mismatch in mismatches:
                LOGGER.warning('Expected: {package}=={expected_version}, found: {version}'.format(**mismatch))
