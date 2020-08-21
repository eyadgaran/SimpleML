'''
Test Suite for persistence to the database
'''

__author__ = 'Elisha Yadgaran'


import unittest
from random import randint

from simpleml.persistables.base_persistable import Persistable
from simpleml.datasets import Dataset
from simpleml.pipelines import Pipeline
from simpleml.models import Model
from simpleml.metrics import Metric


class PersistablePersistenceTests():
    def test_latest_version_retrieved(self):
        # Original Class
        class VersionTestClass(Persistable):
            __tablename__ = 'version_tests'

            def _hash(self):
                return 132435465

        VersionTestClass.__table__.create()

        versions = randint(100, 200)
        for i in range(versions):
            VersionTestClass(name='version_test').save()

        new_class = VersionTestClass(name='version_test')
        self.assertEqual(new_class._get_latest_version(), versions + 1)

    def test_persistence():
        pass


class DatasetPersistenceTests():
    pass


class PipelinePersistenceTests():
    def test_persistence(self):
        '''
        Assert loaded object is the same as original
        '''
        pipeline = Pipeline()
        pipeline.add_dataset(Dataset())
        pipeline.fit()
        pipeline.save()

        pipeline2 = Pipeline.where(
            name=pipeline.name,
            version=pipeline.version
        ).first()

        pipeline2.load()
        assert(pipeline.external_pipeline == pipeline2.external_pipeline)

    def test_dataset_doesnt_load_externals_by_default(self):
        '''
        Start with db record and call load
        '''
        pass


class ModelPersistenceTests():
    pass


class MetricPersistenceTests():
    pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
