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


# class BaseSaveTests(with_metaclass(ABCMeta, object)):
#     '''
#     '''
#     # @abstractmethod
#     # def test_saving(self):
#     #     pass
#     #
#     # @abstractmethod
#     # def test_loading(self):
#     #     pass
#
#
# class DatabaseTableSaveTests(BaseSaveTests, DatabaseTableSaveMixin, unittest.TestCase):
#     @skip_unless_dialect('postgresql')
#     def test_missing_data_casting(self):
#         '''
#         Tests that missing values in the dataframe can still use the postgres \copy
#         command
#         Failure usually with psycopg2.DataError("Missing Value for Column...")
#         '''
#         df = pd.DataFrame([
#             {'a': 123, 'b': None},
#             {'a': None, 'b': 456}
#         ])
#
#         # Save and load
#         engine = DatasetStorageSqlalchemy.metadata.bind
#         self.df_to_sql(engine, df, 'null_test')
#         loaded = pd.read_sql_query('select * from null_test', con=engine)
#         self.assertTrue(df.equals(loaded))
#
#     @skip_unless_dialect('postgresql')
#     def test_handling_line_breaks(self):
#         '''
#         Similar to missing data, line breaks will break copy command
#         Note: column data types will be cast for uniformity so expect casting
#         with mixed types
#         '''
#         df = pd.DataFrame([
#             {'a': 123, 'b': 'agdjldjf\ndf\t\rdf\n'},
#             {'a': None, 'b': 456}
#         ])
#         expected_df = pd.DataFrame([
#             {'a': 123, 'b': 'agdjldjf\ndf\t\rdf\n'},
#             {'a': None, 'b': '456'}
#         ])
#
#         # Save and load
#         engine = DatasetStorageSqlalchemy.metadata.bind
#         self.df_to_sql(engine, df, 'new_line_test')
#         loaded = pd.read_sql_query('select * from new_line_test', con=engine)
#         self.assertFalse(df.equals(loaded))
#         self.assertTrue(expected_df.equals(loaded))
#
#     @skip_unless_dialect('postgresql')
#     def test_dataframe_persistence(self):
#         '''
#         Tests that sql representation of data is equivalent
#         to DataFrame
#         '''
#         df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
#         dataset = Dataset(name='table_saving_test', save_patterns={'dataset': ['database_table']})
#         dataset._external_file = df
#         dataset.save()
#
#         dataset2 = Dataset.where(
#             name=dataset.name,
#             version=dataset.version
#         ).first()
#
#         dataset2.load()
#
#         assert(dataset2.dataframe.equals(dataset.dataframe))
#         assert dataset2.dataframe.equals(df)
#         # .where((pd.notnull(dataset.dataframe)), None).reset_index(drop=True)
#


if __name__ == '__main__':
    unittest.main(verbosity=2)
