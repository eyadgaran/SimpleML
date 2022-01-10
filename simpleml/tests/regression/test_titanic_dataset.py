'''
Regression test with titanic dataset
'''

__author__ = 'Elisha Yadgaran'


import unittest
from os.path import join

from simpleml.registries import FILEPATH_REGISTRY
from simpleml.tests.utils import (ARTIFACTS_PATH, DATABASES_PATH,
                                  assert_data_container_equal)
from simpleml.utils.initialization import Database
from simpleml.utils.scoring.load_persistable import PersistableLoader
from simpleml.utils.training.create_persistable import DatasetCreator

dataset_kwargs_template = {
    'project': 'regression-tests',
    'label_columns': ['Survived'],
    'filepath': '../data/mock/titanic.csv',  # keep relative for consistent hash
    'format': 'csv',
    'squeeze_return': True,
}


class TitanicRegressionTest(unittest.TestCase):
    '''
    End to end regression test for fully functional project.
    Uses kaggle Titanic project as a worked example

    Validates behavior and compatibility has not regressed (first by executing
    latter by comparing to fixed references)
    '''
    @classmethod
    def setUpClass(cls):
        '''
        Establish connection to regression db
        '''
        cls.db = Database(uri=f"sqlite:///{join(DATABASES_PATH, 'titanic-regression.db')}")
        cls.db.initialize()

    def get_regression_artifact(self, persistable_type, **filters):
        with FILEPATH_REGISTRY.context_register('filestore', ARTIFACTS_PATH):
            persistable = getattr(PersistableLoader, f'load_{persistable_type}')(**filters)
            persistable.load(load_externals=True)
            return persistable

    def compare_hashes(self, new, old):
        '''
        Assert content specification has been unchanged
        '''
        self.assertEqual(new.hash_, old.hash_)

    def compare_datasets(self, new, old):
        self.compare_hashes(new, old)
        assert_data_container_equal(new._external_file, old._external_file)

    def test_dask_datasets(self):
        registered_name = 'DaskFileBasedDataset'
        name = 'titanic-regression-{test_name}'
        save_patterns = {'dataset': ['{save_pattern}']}

        for save_pattern in ['dask_disk_json', 'dask_disk_csv', 'dask_disk_parquet']:
            with self.subTest('Dask Dataset Regression with Titanic', save_pattern=save_pattern):
                save_patterns['dataset'] = [save_pattern]
                regression_dataset = self.get_regression_artifact('dataset', name=name.format(test_name=save_pattern))
                with FILEPATH_REGISTRY.context_register('filestore', ARTIFACTS_PATH):
                    new_dataset = DatasetCreator.create(registered_name=registered_name,
                                                        name=name.format(test_name=f'{save_pattern}_new'),
                                                        save_patterns=save_patterns,
                                                        **dataset_kwargs_template)

                self.compare_datasets(new_dataset, regression_dataset)

    def test_pandas_datasets(self):
        registered_name = 'PandasFileBasedDataset'
        name = 'titanic-regression-{test_name}'
        save_patterns = {'dataset': ['{save_pattern}']}

        for save_pattern in ['pandas_disk_json', 'pandas_disk_csv', 'pandas_disk_parquet']:
            with self.subTest('Pandas Dataset Regression with Titanic', save_pattern=save_pattern):
                save_patterns['dataset'] = [save_pattern]
                regression_dataset = self.get_regression_artifact('dataset', name=name.format(test_name=save_pattern))
                with FILEPATH_REGISTRY.context_register('filestore', ARTIFACTS_PATH):
                    new_dataset = DatasetCreator.create(registered_name=registered_name,
                                                        name=name.format(test_name=f'{save_pattern}_new'),
                                                        save_patterns=save_patterns,
                                                        **dataset_kwargs_template)

                self.compare_datasets(new_dataset, regression_dataset)


if __name__ == '__main__':
    unittest.main(verbosity=2)
