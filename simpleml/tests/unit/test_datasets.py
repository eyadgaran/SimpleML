'''
Dataset related tests
'''

__author__ = 'Elisha Yadgaran'


import itertools
import unittest

import numpy as np
import pandas as pd
from simpleml.datasets.base_dataset import Dataset
from simpleml.datasets.dask import (BaseDaskDataset, DaskFileBasedDataset,
                                    DaskPipelineDataset)
from simpleml.datasets.numpy import BaseNumpyDataset, NumpyPipelineDataset
from simpleml.datasets.pandas import (BasePandasDataset,
                                      PandasFileBasedDataset,
                                      PandasPipelineDataset)
from simpleml.datasets.pandas.base import DATAFRAME_SPLIT_COLUMN
from simpleml.imports import dd, ddDataFrame, ddSeries
from simpleml.tests.utils import assert_data_container_equal
from simpleml.utils.errors import DatasetError


class AbstractDatasetTests(unittest.TestCase):
    '''
    Tests for abstract mixin class
    '''

    def test_abstract_methods(self):
        dataset = Dataset()

        with self.assertRaises(NotImplementedError):
            dataset.build_dataframe()

        with self.assertRaises(NotImplementedError):
            dataset.dataframe

        with self.assertRaises(AttributeError):
            dataset._dataframe

        with self.assertRaises(NotImplementedError):
            dataset.X

        with self.assertRaises(NotImplementedError):
            dataset.y

        with self.assertRaises(NotImplementedError):
            dataset.get('', '')

        with self.assertRaises(NotImplementedError):
            dataset.get_feature_names()

        with self.assertRaises(NotImplementedError):
            dataset.get_split('any')

        with self.assertRaises(NotImplementedError):
            dataset.get_split_names()


class BaseDatasetTests(unittest.TestCase):
    def test_section_column_handling(self):
        '''
        Datasets can be configured with arbitrary columns
        '''
        with self.subTest('Y labels'):
            section = 'y'
            columns = ['a', 'b']
            dataset = Dataset(label_columns=columns)
            self.assertEqual(dataset.get_section_columns(section), columns)

        with self.subTest('Random section'):
            section = 'abc'
            columns = ['a', 'b']
            dataset = Dataset(other_named_split_sections={section: columns})
            self.assertEqual(dataset.get_section_columns(section), columns)

        with self.subTest('nonexistent section'):
            section = 'abc'
            columns = []
            dataset = Dataset()
            self.assertEqual(dataset.get_section_columns(section), columns)


class _DatasetTestHelper(object):
    '''
    All mixins should run these tests with the appropriate setups per class

    - self.dummy_dataset
    - self.y_equality_function
    - self._data
    - self.expected_dataframe
    - self.expected_x
    - self.expected_y
    - self.expected_train_dataframe
    - self.expected_train_x
    - self.expected_train_y
    '''

    '''
    property tests
    '''

    def test_dataframe_set_validation(self):
        '''
        Check requirement
        '''
        dataset = self.dummy_dataset
        with self.assertRaises(DatasetError):
            dataset.dataframe = 'blah'

        df = pd.DataFrame()
        if issubclass(self.dataset_cls, BaseDaskDataset):
            df = dd.from_pandas(df, npartitions=2)

        dataset.dataframe = df

    def test_y(self):
        '''
        Test property wrapper - same as get y
        '''
        dataset = self.dummy_dataset
        get_y = dataset.get(column='y', split=None)
        assert_data_container_equal(self.expected_y, get_y)
        assert_data_container_equal(self.expected_y, dataset.y)

    def test_x(self):
        '''
        Test property wrapper - same as get X
        '''
        dataset = self.dummy_dataset
        get_x = dataset.get(column='X', split=None)
        assert_data_container_equal(self.expected_x, get_x)
        assert_data_container_equal(self.expected_x, dataset.X)

    def test_dataframe(self):
        '''
        Should return a copy of the full dataset
        '''
        dataset = self.dummy_dataset
        get_dataframe = dataset._dataframe
        assert_data_container_equal(self._data, get_dataframe)

    '''
    get tests
    '''

    def test_get_nonexistent_column_error(self):
        '''
        Should raise an error
        '''
        dataset = self.dummy_dataset
        with self.assertRaises(ValueError):
            dataset.get(column='other', split=None)

    def test_missing_split_column_error(self):
        '''
        Attempt to query a split from a dataframe without the split column
        Would otherwise throw a KeyError
        '''
        dataset = self.dummy_dataset
        dataset._external_file = dataset._external_file.drop(DATAFRAME_SPLIT_COLUMN, axis=1)
        with self.assertRaises(DatasetError):
            dataset.get(column='X', split='Nonsense')

    def test_get_nonexistent_split(self):
        '''
        Should return an empty frame
        '''
        dataset = self.dummy_dataset
        X = dataset.get(column='X', split='NONSENSE')
        y = dataset.get(column='y', split='NONSENSE')
        data = dataset.get(column=None, split='NONSENSE')

        if isinstance(data, ddDataFrame):
            # head calls return a pd.DataFrame/series
            assert_data_container_equal(X.compute(), self.expected_x.head(0))
            assert_data_container_equal(y.compute(), self.expected_y.head(0))
            assert_data_container_equal(data.compute(), self.expected_dataframe.head(0))
        else:
            assert_data_container_equal(X, self.expected_x.head(0))
            assert_data_container_equal(y, self.expected_y.head(0))
            assert_data_container_equal(data, self.expected_dataframe.head(0))

    def test_get_with_split(self):
        '''
        Should return df slices
        '''
        dataset = self.dummy_dataset
        X = dataset.get(column='X', split='TRAIN')
        y = dataset.get(column='y', split='TRAIN')
        data = dataset.get(column=None, split='TRAIN')

        assert_data_container_equal(self.expected_train_x, X)
        assert_data_container_equal(self.expected_train_y, y)
        assert_data_container_equal(self.expected_train_dataframe, data)

    def test_get_with_null_parameters(self):
        '''
        Should return all columns and rows except the split column
        '''
        dataset = self.dummy_dataset
        get_dataframe = dataset.get(column=None, split=None)
        assert_data_container_equal(self.expected_dataframe, get_dataframe)

    '''
    references
    '''

    def test_dataframe_reference(self):
        '''
        Calling Dataset.dataframe returns copy
        '''
        dataset = self.dummy_dataset
        self.assertNotEqual(id(dataset._dataframe), id(dataset._external_file))

    def test_dataframe_mutability(self):
        '''
        Test mutating dataframe doesnt affect raw data
        '''
        dataset = self.dummy_dataset
        copy = dataset._dataframe
        try:
            copy.drop(DATAFRAME_SPLIT_COLUMN, axis=1, inplace=True)
        except TypeError:  # dask
            copy = copy.drop(DATAFRAME_SPLIT_COLUMN, axis=1)
        with self.assertRaises(AssertionError):
            assert_data_container_equal(dataset._dataframe, copy)

    def test_get_X_mutability(self):
        '''
        Pandas dataframes often return copies and views for efficiency.
        Views can cause inplace mutations to propagate back to the original
        dataframe. That is not allowed to maintain the integrity of the persisted
        data

        Tests for:
        - memory pointers (object id)
        - df._is_copy is not None (weakref when attached to a parent df)
        - df._is_view is False (True for certain slices)
        '''
        dataset = self.dummy_dataset

        if issubclass(self.dataset_cls, BaseDaskDataset):  # dask
            unmodified_copy = dataset.dataframe.copy()
        else:
            unmodified_copy = dataset.dataframe.copy(deep=True)

        for column, split in itertools.product(
            ['X'],
            ['TRAIN', 'TEST', 'VALIDATION', None]
        ):
            with self.subTest(column=column, split=split):
                copy = dataset.get(column=column, split=split)

                try:
                    # Test for pandas references
                    self.assertIsNone(copy._is_copy)

                    # Not fully understood behavior causes pandas to return views for certain
                    # operations that morph into copies when modified (appears subject to mem optimizations)
                    self.assertFalse(copy._is_view)

                    # Modify copy and see if the source changed
                    # dask doesnt support item assignment
                    copy.loc[1, 'a'] = 9876

                    with self.assertRaises(AssertionError):
                        assert_data_container_equal(copy, dataset.get(column=column, split=split))

                except AttributeError:
                    pass

                assert_data_container_equal(dataset.dataframe, unmodified_copy)

                # id pointer
                self.assertNotEqual(id(dataset._external_file), id(copy))
                self.assertNotEqual(id(dataset._external_file), id(dataset.dataframe))
                self.assertNotEqual(id(dataset._external_file), id(unmodified_copy))
                self.assertNotEqual(id(dataset.dataframe), id(copy))
                self.assertNotEqual(id(unmodified_copy), id(copy))

    def test_feature_names(self):
        '''
        Only support X as the initial feature
        '''
        self.assertEqual(self.dummy_dataset.get_feature_names(), ['a', 'b'])

    '''
    logic
    '''

    def test_hash_consistency(self):
        '''
        Ensure hash equivalence as long as underlying dataset
        hasnt changed
        '''
        df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
        df_shuffled = df.sample(frac=1)
        self.assertNotEqual(df.index.tolist(), df_shuffled.index.tolist())

        dataset1 = self.dummy_dataset
        dataset1._external_file = df

        dataset2 = self.dummy_dataset
        dataset2._external_file = df_shuffled

        self.assertEqual(dataset1._hash(), dataset2._hash())


class BasePandasDatasetTests(unittest.TestCase, _DatasetTestHelper):
    '''
    Tests for the pandas related functionality
    '''
    dataset_cls = BasePandasDataset

    @property
    def expected_dataframe(self):
        return pd.DataFrame(
            [
                {'a': 1, 'b': 2, 'label': 3},
                {'a': 11, 'b': 22, 'label': 33},
                {'a': 111, 'b': 222, 'label': 333}
            ],
            index=[10, 20, 30]
        )

    @property
    def expected_x(self):
        return pd.DataFrame(
            [
                {'a': 1, 'b': 2},
                {'a': 11, 'b': 22},
                {'a': 111, 'b': 222}
            ],
            index=[10, 20, 30]
        )

    @property
    def expected_y(self):
        '''
        Y values are expected to be squeezed
        '''
        return pd.DataFrame(
            [
                {'label': 3},
                {'label': 33},
                {'label': 333}
            ],
            index=[10, 20, 30]
        )

    @property
    def expected_train_dataframe(self):
        return pd.DataFrame(
            [
                {'a': 1, 'b': 2, 'label': 3},
            ],
            index=[10]
        )

    @property
    def expected_train_x(self):
        return pd.DataFrame(
            [
                {'a': 1, 'b': 2},
            ],
            index=[10]
        )

    @property
    def expected_train_y(self):
        return pd.DataFrame(
            [
                {'label': 3},
            ],
            index=[10]
        )

    @property
    def _data(self):
        return pd.DataFrame(
            [
                {'a': 1, 'b': 2, 'label': 3, DATAFRAME_SPLIT_COLUMN: 'TRAIN'},
                {'a': 11, 'b': 22, 'label': 33, DATAFRAME_SPLIT_COLUMN: 'VALIDATION'},
                {'a': 111, 'b': 222, 'label': 333, DATAFRAME_SPLIT_COLUMN: 'TEST'},
            ],
            index=[10, 20, 30]
        )

    @property
    def dummy_dataset(self):
        dataset = self.dataset_cls(label_columns=['label'])
        dataset.dataframe = self._data
        return dataset


class SingleLabelPandasDatasetTests(BasePandasDatasetTests):
    '''
    Same tests but overload labels to not squeeze to a numpy array
    '''
    dataset_cls = BasePandasDataset

    @property
    def expected_y(self):
        '''
        Y values are expected to be squeezed
        '''
        return pd.Series(
            [
                3,
                33,
                333
            ],
            name='label',
            index=[10, 20, 30]
        )

    @property
    def expected_train_y(self):
        return pd.Series(
            [
                3,
            ],
            name='label',
            index=[10]
        )

    @property
    def dummy_dataset(self):
        dataset = self.dataset_cls(label_columns=['label'], squeeze_return=True)
        dataset.dataframe = self._data
        return dataset


class MultiLabelPandasDatasetTests(BasePandasDatasetTests):
    '''
    Same tests but overload labels to not squeeze to a numpy array
    '''
    dataset_cls = BasePandasDataset

    @property
    def expected_dataframe(self):
        return pd.DataFrame(
            [
                {'a': 1, 'b': 2, 'label1': 3, 'label2': 4},
                {'a': 11, 'b': 22, 'label1': 33, 'label2': 44},
                {'a': 111, 'b': 222, 'label1': 333, 'label2': 444}
            ],
            index=[10, 20, 30]
        )

    @property
    def expected_y(self):
        return pd.DataFrame(
            [
                {'label1': 3, 'label2': 4},
                {'label1': 33, 'label2': 44},
                {'label1': 333, 'label2': 444}
            ],
            index=[10, 20, 30]
        )

    @property
    def expected_train_dataframe(self):
        return pd.DataFrame(
            [
                {'a': 1, 'b': 2, 'label1': 3, 'label2': 4},
            ],
            index=[10]
        )

    @property
    def expected_train_y(self):
        return pd.DataFrame(
            [
                {'label1': 3, 'label2': 4},
            ],
            index=[10]
        )

    @property
    def _data(self):
        return pd.DataFrame(
            [
                {'a': 1, 'b': 2, 'label1': 3, 'label2': 4, DATAFRAME_SPLIT_COLUMN: 'TRAIN'},
                {'a': 11, 'b': 22, 'label1': 33, 'label2': 44, DATAFRAME_SPLIT_COLUMN: 'VALIDATION'},
                {'a': 111, 'b': 222, 'label1': 333, 'label2': 444, DATAFRAME_SPLIT_COLUMN: 'TEST'},
            ],
            index=[10, 20, 30]
        )

    @property
    def dummy_dataset(self):
        dataset = self.dataset_cls(label_columns=['label1', 'label2'], sqeeze_return=False)
        dataset.dataframe = self._data
        return dataset


class CSVFileBasedPandasDatasetTests(SingleLabelPandasDatasetTests):
    dataset_cls = PandasFileBasedDataset

    @property
    def dummy_dataset(self):
        dataset = self.dataset_cls(filepath='', format='csv', label_columns=['label'], squeeze_return=True)
        dataset.dataframe = self._data
        return dataset


class JSONFileBasedPandasDatasetTests(SingleLabelPandasDatasetTests):
    dataset_cls = PandasFileBasedDataset

    @property
    def dummy_dataset(self):
        dataset = self.dataset_cls(filepath='', format='json', label_columns=['label'], squeeze_return=True)
        dataset.dataframe = self._data
        return dataset


class ParquetFileBasedPandasDatasetTests(SingleLabelPandasDatasetTests):
    dataset_cls = PandasFileBasedDataset

    @property
    def dummy_dataset(self):
        dataset = self.dataset_cls(filepath='', format='parquet', label_columns=['label'], squeeze_return=True)
        dataset.dataframe = self._data
        return dataset


class PipelinePandasDatasetTests(SingleLabelPandasDatasetTests):
    dataset_cls = PandasPipelineDataset

    @property
    def dummy_dataset(self):
        dataset = self.dataset_cls(label_columns=['label'], squeeze_return=True)
        dataset.dataframe = self._data
        return dataset


class BaseDaskDatasetTests(BasePandasDatasetTests):
    '''
    Tests for the dask related functionality
    '''
    dataset_cls = BaseDaskDataset

    @property
    def expected_dataframe(self):
        return dd.from_pandas(super().expected_dataframe, npartitions=2)

    @property
    def expected_x(self):
        return dd.from_pandas(super().expected_x, npartitions=2)

    @property
    def expected_y(self):
        '''
        Y values are expected to be squeezed
        '''
        return dd.from_pandas(super().expected_y, npartitions=2)

    @property
    def expected_train_dataframe(self):
        return dd.from_pandas(super().expected_train_dataframe, npartitions=2)

    @property
    def expected_train_x(self):
        return dd.from_pandas(super().expected_train_x, npartitions=2)

    @property
    def expected_train_y(self):
        return dd.from_pandas(super().expected_train_y, npartitions=2)

    @property
    def _data(self):
        return dd.from_pandas(super()._data, npartitions=2)


class SingleLabelDaskDatasetTests(SingleLabelPandasDatasetTests):
    '''
    Same tests but overload labels to not squeeze to a numpy array
    '''
    dataset_cls = BaseDaskDataset

    @property
    def expected_dataframe(self):
        return dd.from_pandas(super().expected_dataframe, npartitions=2)

    @property
    def expected_x(self):
        return dd.from_pandas(super().expected_x, npartitions=2)

    @property
    def expected_y(self):
        '''
        Y values are expected to be squeezed
        '''
        return dd.from_pandas(super().expected_y, npartitions=2)

    @property
    def expected_train_dataframe(self):
        return dd.from_pandas(super().expected_train_dataframe, npartitions=2)

    @property
    def expected_train_x(self):
        return dd.from_pandas(super().expected_train_x, npartitions=2)

    @property
    def expected_train_y(self):
        return dd.from_pandas(super().expected_train_y, npartitions=2)

    @property
    def _data(self):
        return dd.from_pandas(super()._data, npartitions=2)


class MultiLabelDaskDatasetTests(MultiLabelPandasDatasetTests):
    '''
    Same tests but overload labels to not squeeze to a numpy array
    '''
    dataset_cls = BaseDaskDataset

    @property
    def expected_dataframe(self):
        return dd.from_pandas(super().expected_dataframe, npartitions=2)

    @property
    def expected_x(self):
        return dd.from_pandas(super().expected_x, npartitions=2)

    @property
    def expected_y(self):
        '''
        Y values are expected to be squeezed
        '''
        return dd.from_pandas(super().expected_y, npartitions=2)

    @property
    def expected_train_dataframe(self):
        return dd.from_pandas(super().expected_train_dataframe, npartitions=2)

    @property
    def expected_train_x(self):
        return dd.from_pandas(super().expected_train_x, npartitions=2)

    @property
    def expected_train_y(self):
        return dd.from_pandas(super().expected_train_y, npartitions=2)

    @property
    def _data(self):
        return dd.from_pandas(super()._data, npartitions=2)


class CSVFileBasedDaskDatasetTests(SingleLabelDaskDatasetTests):
    dataset_cls = DaskFileBasedDataset

    @property
    def dummy_dataset(self):
        dataset = self.dataset_cls(filepath='', format='csv', label_columns=['label'], squeeze_return=True)
        dataset.dataframe = self._data
        return dataset


class JSONFileBasedDaskDatasetTests(SingleLabelDaskDatasetTests):
    dataset_cls = DaskFileBasedDataset

    @property
    def dummy_dataset(self):
        dataset = self.dataset_cls(filepath='', format='json', label_columns=['label'], squeeze_return=True)
        dataset.dataframe = self._data
        return dataset


class ParquetFileBasedDaskDatasetTests(SingleLabelDaskDatasetTests):
    dataset_cls = DaskFileBasedDataset

    @property
    def dummy_dataset(self):
        dataset = self.dataset_cls(filepath='', format='parquet', label_columns=['label'], squeeze_return=True)
        dataset.dataframe = self._data
        return dataset


class PipelineDaskDatasetTests(SingleLabelDaskDatasetTests):
    dataset_cls = DaskPipelineDataset

    @property
    def dummy_dataset(self):
        dataset = self.dataset_cls(label_columns=['label'], squeeze_return=True)
        dataset.dataframe = self._data
        return dataset


class BaseNumpyDatasetTests(unittest.TestCase):
    '''
    Some tests for the numpy mixin class
    '''
    @property
    def expected_x(self):
        return 2

    @property
    def expected_y(self):
        return 3

    @property
    def dummy_dataset(self):
        dataset = BaseNumpyDataset(label_columns=['label'])
        dataset._external_file = {'X': self.expected_x, 'label': self.expected_y}
        return dataset

    @property
    def dummy_split_dataset(self):
        dataset = BaseNumpyDataset(label_columns=['label'])
        dataset._external_file = {
            'TRAIN': {'X': self.expected_x, 'label': self.expected_y}
        }
        return dataset

    def test_get_column_error(self):
        dataset = self.dummy_dataset
        with self.assertRaises(ValueError):
            dataset.get(column='other', split=None)

    def test_get_no_split(self):
        dataset = self.dummy_dataset
        X = dataset.get(column='X', split=None)
        y = dataset.get(column='y', split=None)
        self.assertEqual(self.expected_x, X)
        self.assertEqual(self.expected_y, y)

    def test_get_with_split(self):
        dataset = self.dummy_split_dataset
        X = dataset.get(column='X', split='TRAIN')
        y = dataset.get(column='y', split='TRAIN')
        self.assertEqual(self.expected_x, X)
        self.assertEqual(self.expected_y, y)

    def test_y(self):
        dataset = self.dummy_dataset
        get_y = dataset.get(column='y', split=None)
        self.assertEqual(self.expected_y, get_y)
        self.assertEqual(self.expected_y, dataset.y)

    def test_x(self):
        dataset = self.dummy_dataset
        get_x = dataset.get(column='X', split=None)
        self.assertEqual(self.expected_x, get_x)
        self.assertEqual(self.expected_x, dataset.X)

    def test_feature_names(self):
        '''
        Only support X as the initial feature
        '''
        self.assertEqual(self.dummy_dataset.get_feature_names(), ['X'])
        self.assertEqual(self.dummy_split_dataset.get_feature_names(), ['X'])


class PipelineNumpyDatasetTests(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
