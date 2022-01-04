'''
Integration tests for save patterns
Includes available stateful connections (database) and full class implementations
'''

__author__ = 'Elisha Yadgaran'


import os
import unittest

from simpleml.datasets.dask import BaseDaskDataset
from simpleml.imports import dd
from simpleml.save_patterns.serializers.dask import DaskPersistenceMethods
from simpleml.tests.utils import MOCK_PATH, assert_data_container_equal


class DaskPersistenceTests(unittest.TestCase):
    def generate_dataset(self, save_pattern):
        dataset = BaseDaskDataset(name='dask_integration_tests', label_columns=['Survived'],
                                  squeeze_return=True,
                                  save_patterns={'dataset': [save_pattern]})
        dataset.dataframe = dd.read_csv(os.path.join(MOCK_PATH, 'titanic.csv')).repartition(npartitions=20)
        return dataset

    def test_save_and_load_json(self):
        save_pattern = 'dask_disk_json'
        dataset = self.generate_dataset(save_pattern)
        df = dataset.dataframe.compute()
        # expect a custom index
        df.index.name = DaskPersistenceMethods.INDEX_COLUMN
        dataset.save()
        dataset.load()
        df2 = dataset.dataframe.compute()
        assert_data_container_equal(df, df2)

    def test_save_and_load_csv(self):
        save_pattern = 'dask_disk_csv'
        dataset = self.generate_dataset(save_pattern)
        df = dataset.dataframe.compute()
        # expect a custom index
        df.index.name = DaskPersistenceMethods.INDEX_COLUMN
        dataset.save()
        dataset.load()
        df2 = dataset.dataframe.compute()
        assert_data_container_equal(df, df2)

    def test_save_and_load_parquet(self):
        save_pattern = 'dask_disk_parquet'
        dataset = self.generate_dataset(save_pattern)
        df = dataset.dataframe.compute()
        dataset.save()
        dataset.load()
        df2 = dataset.dataframe.compute()
        assert_data_container_equal(df, df2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
