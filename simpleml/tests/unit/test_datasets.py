'''
Dataset related tests
'''

__author__ = 'Elisha Yadgaran'


import unittest
import numpy as np
import pandas as pd
import itertools
from pandas.testing import assert_frame_equal, assert_series_equal

from simpleml.datasets import PandasDataset, SingleLabelPandasDataset, MultiLabelPandasDataset, NumpyDataset
from simpleml.datasets.base_dataset import Dataset, AbstractDataset
from simpleml.datasets.abstract_mixin import AbstractDatasetMixin
from simpleml.datasets.numpy_mixin import NumpyDatasetMixin
from simpleml.datasets.pandas_mixin import BasePandasDatasetMixin, \
    SingleLabelPandasDatasetMixin, MultiLabelPandasDatasetMixin, DATAFRAME_SPLIT_COLUMN
from simpleml.utils.errors import DatasetError


class AbstractMixinTests(unittest.TestCase):
    '''
    Tests for abstract mixin class
    '''
    @property
    def dummy_dataset(self):
        class TestMixinClass(AbstractDatasetMixin):
            pass
        return TestMixinClass()

    def test_abstract_methods(self):
        dataset = self.dummy_dataset

        with self.assertRaises(NotImplementedError):
            dataset.X

        with self.assertRaises(NotImplementedError):
            dataset.y

        with self.assertRaises(NotImplementedError):
            dataset.get('', '')

        with self.assertRaises(NotImplementedError):
            dataset.get_feature_names()


class _PandasTestHelper(object):
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
        Check requirement for pd.DataFrame
        '''
        dataset = self.dummy_dataset
        with self.assertRaises(DatasetError):
            dataset._dataframe = 'blah'

        dataset._dataframe = pd.DataFrame()

    def test_y(self):
        '''
        Test property wrapper - same as get y
        '''
        dataset = self.dummy_dataset
        get_y = dataset.get(column='y', split=None)
        self.y_equality_function(self.expected_y, get_y)
        self.y_equality_function(self.expected_y, dataset.y)

    def test_x(self):
        '''
        Test property wrapper - same as get X
        '''
        dataset = self.dummy_dataset
        get_x = dataset.get(column='X', split=None)
        assert_frame_equal(self.expected_x, get_x)
        assert_frame_equal(self.expected_x, dataset.X)

    def test_dataframe(self):
        '''
        Should return a copy of the full dataset
        '''
        dataset = self.dummy_dataset
        get_dataframe = dataset._dataframe
        assert_frame_equal(self._data, get_dataframe)

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
        dataset._external_file.drop(DATAFRAME_SPLIT_COLUMN, axis=1, inplace=True)
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
        assert_frame_equal(X, self.expected_x.head(0))
        self.y_equality_function(y, self.expected_y.head(0))
        assert_frame_equal(data, self.expected_dataframe.head(0))

    def test_get_with_split(self):
        '''
        Should return df slices
        '''
        dataset = self.dummy_dataset
        X = dataset.get(column='X', split='TRAIN')
        y = dataset.get(column='y', split='TRAIN')
        data = dataset.get(column=None, split='TRAIN')

        assert_frame_equal(self.expected_train_x, X)
        self.y_equality_function(self.expected_train_y, y)
        assert_frame_equal(self.expected_train_dataframe, data)

    def test_get_with_null_parameters(self):
        '''
        Should return all columns and rows except the split column
        '''
        dataset = self.dummy_dataset
        get_dataframe = dataset.get(column=None, split=None)
        assert_frame_equal(self.expected_dataframe, get_dataframe)

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
        copy.drop(DATAFRAME_SPLIT_COLUMN, axis=1, inplace=True)
        with self.assertRaises(AssertionError):
            assert_frame_equal(dataset._dataframe, copy)

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
        unmodified_copy = dataset.dataframe.copy(deep=True)

        for column, split in itertools.product(
            ['X'],
            ['TRAIN', 'TEST', 'VALIDATION', None]
        ):
            with self.subTest(column=column, split=split):
                copy = dataset.get(column=column, split=split)

                # Test for pandas references
                self.assertIsNone(copy._is_copy)

                # Not fully understood behavior causes pandas to return views for certain
                # operations that morph into copies when modified (appears subject to mem optimizations)
                self.assertFalse(copy._is_view)

                # Modify copy and see if the source changed
                copy.loc[1, 'a'] = 9876

                assert_frame_equal(dataset.dataframe, unmodified_copy)
                with self.assertRaises(AssertionError):
                    assert_frame_equal(copy, dataset.get(column=column, split=split))

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


class BasePandasMixinTests(unittest.TestCase, _PandasTestHelper):
    '''
    Tests for the pandas related functionality
    '''

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
    def y_equality_function(self):
        return assert_frame_equal

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
        class TestMixinClass(BasePandasDatasetMixin):
            _external_file = self._data
            label_columns = ['label']

            @property
            def dataframe(self):
                return self._dataframe

        return TestMixinClass()


class PandasDatasetTests(BasePandasDatasetMixin):
    @property
    def dummy_dataset(self):
        dataset = PandasDataset(label_columns=['label'])
        dataset._external_file = self._data
        return dataset


class SingleLabelPandasMixinTests(unittest.TestCase, _PandasTestHelper):
    '''
    Same tests but overload labels to not squeeze to a numpy array
    '''
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
    def y_equality_function(self):
        return assert_series_equal

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
        return pd.Series(
            [
                3,
            ],
            name='label',
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
        class TestMixinClass(SingleLabelPandasDatasetMixin):
            _external_file = self._data
            label_columns = ['label']

            @property
            def dataframe(self):
                return self._dataframe

        return TestMixinClass()


class SingleLabelPandasDatasetTests(SingleLabelPandasMixinTests):
    @property
    def dummy_dataset(self):
        dataset = SingleLabelPandasDataset(label_columns=['label'])
        dataset._external_file = self._data
        return dataset


class MultiLabelPandasMixinTests(unittest.TestCase, _PandasTestHelper):
    '''
    Same tests but overload labels to not squeeze to a numpy array
    '''

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
        return pd.DataFrame(
            [
                {'label1': 3, 'label2': 4},
                {'label1': 33, 'label2': 44},
                {'label1': 333, 'label2': 444}
            ],
            index=[10, 20, 30]
        )

    @property
    def y_equality_function(self):
        return assert_frame_equal

    @property
    def expected_train_dataframe(self):
        return pd.DataFrame(
            [
                {'a': 1, 'b': 2, 'label1': 3, 'label2': 4},
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
        class TestMixinClass(MultiLabelPandasDatasetMixin):
            _external_file = self._data
            label_columns = ['label1', 'label2']

            @property
            def dataframe(self):
                return self._dataframe

        return TestMixinClass()


class MultiLabelPandasDatasetTests(MultiLabelPandasMixinTests):
    @property
    def dummy_dataset(self):
        dataset = MultiLabelPandasDataset(label_columns=['label1', 'label2'])
        dataset._external_file = self._data
        return dataset


class NumpyMixinTests(unittest.TestCase):
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
        class TestMixinClass(NumpyDatasetMixin):
            dataframe = {'X': self.expected_x, 'label': self.expected_y}
            label_columns = ['label']

        return TestMixinClass()

    @property
    def dummy_split_dataset(self):
        class TestMixinClass(NumpyDatasetMixin):
            dataframe = {
                'TRAIN': {'X': self.expected_x, 'label': self.expected_y}
            }
            label_columns = ['label']

        return TestMixinClass()

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


class DatasetTests(unittest.TestCase):
    def test_build_dataframe_called(self):
        pass

    def test_hash_consistency(self):
        '''
        Ensure hash equivalence as long as underlying dataset
        hasnt changed
        '''
        df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list('ABCD'))
        df_shuffled = df.sample(frac=1)
        self.assertNotEqual(df.index.tolist(), df_shuffled.index.tolist())

        dataset1 = Dataset()
        dataset1._external_file = df

        dataset2 = Dataset()
        dataset2._external_file = df_shuffled

        self.assertEqual(dataset1._hash(), dataset2._hash())

    def test_hash(self):
        '''
        Compare hash to hard-coded precomputed hash
        '''


class ImplementationTests(unittest.TestCase):
    '''
    Implementation tests for new dataset objects - via inheritance (custom `build_dataframe()`)
    or helper method, like `load_csv()`
    '''
    pass


class PreprocessedImplementationTests(unittest.TestCase):
    '''
    Implementation tests for datasets where a dataset pipeline was used
    first to transform a raw dataset
    '''

    def pandas_dataframe_with_no_split_pipeline(self):
        pass

    def pandas_dataframe_with_explicit_split_pipeline(self):
        pass

    def pandas_dataframe_with_random_split_pipeline(self):
        pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
