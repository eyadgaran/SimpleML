'''
Pipeline Validation Splitting related tests
'''

__author__ = 'Elisha Yadgaran'


import unittest

import numpy as np
import pandas as pd
from simpleml.datasets.dataset_splits import Split
from simpleml.imports import dd
from simpleml.pipelines.validation_split_mixins import (
    ChronologicalSplitMixin, ExplicitSplitMixin, KFoldSplitMixin,
    RandomSplitMixin, SplitMixin)
from simpleml.tests.utils import assert_data_container_equal


class SplitMixinTests(unittest.TestCase):
    def test_abstract_behavior(self):
        with self.assertRaises(TypeError):
            SplitMixin()


class RandomSplitMixinTests(unittest.TestCase):
    class TestRandomSplitMixin(RandomSplitMixin):
        config = {}

    class MockDataset(object):
        def __init__(self, split):
            self.split = split

        @property
        def X(self):
            return self.split.X

        def get_split(self, **kwargs):
            return self.split

    def test_index_generation(self):
        with self.subTest('Pandas DataFrame mixed'):
            index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
            data = pd.DataFrame(range(10), index=index)
            self.assertEqual(RandomSplitMixin.get_index(data), index)

        with self.subTest('Pandas DataFrame linear'):
            data = pd.DataFrame(range(10))
            self.assertEqual(RandomSplitMixin.get_index(data), list(range(10)))

        with self.subTest('Pandas series linear'):
            data = pd.Series(range(10))
            self.assertEqual(RandomSplitMixin.get_index(data), list(range(10)))

        with self.subTest('Pandas series mixed'):
            index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
            data = pd.Series(range(10), index=index)
            self.assertEqual(RandomSplitMixin.get_index(data), index)

        with self.subTest('numpy'):
            data = np.array(range(10))
            self.assertEqual(RandomSplitMixin.get_index(data), list(range(10)))

        with self.subTest('dask DataFrame mixed'):
            index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
            data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=3)
            self.assertEqual(RandomSplitMixin.get_index(data), sorted(index))

        with self.subTest('dask DataFrame linear'):
            data = dd.from_pandas(pd.DataFrame(range(10)), npartitions=3)
            self.assertEqual(RandomSplitMixin.get_index(data), list(range(10)))

        with self.subTest('dask series linear'):
            data = dd.from_pandas(pd.Series(range(10)), npartitions=3)
            self.assertEqual(RandomSplitMixin.get_index(data), list(range(10)))

        with self.subTest('dask series mixed'):
            index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
            data = dd.from_pandas(pd.Series(range(10), index=index), npartitions=3)
            self.assertEqual(RandomSplitMixin.get_index(data), sorted(index))

    '''
    Pandas
    '''

    def test_default_all_splits_with_pandas_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=None,
            test_size=None,
            validation_size=None,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            pd.DataFrame(range(10), index=index)
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_default_all_splits_with_pandas_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=None,
            test_size=None,
            validation_size=None,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            pd.DataFrame(range(10), index=index)
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    '''
    Counts + Pandas
    '''

    def test_all_splits_with_pandas_and_counts_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=3,
            test_size=3,
            validation_size=4,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            pd.DataFrame([{0: 0}, {0: 1}, {0: 2}], index=[1, 3, 4])
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            pd.DataFrame([{0: 3}, {0: 4}, {0: 5}, {0: 6}], index=[5, 8, 19, 43])
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            pd.DataFrame([{0: 7}, {0: 8}, {0: 9}], index=[23, 77, 29])
        )

    def test_all_splits_with_pandas_and_counts_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=3,
            test_size=3,
            validation_size=4,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            pd.DataFrame([{0: 4}, {0: 9}, {0: 1}], index=[8, 29, 3])
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            pd.DataFrame([{0: 3}, {0: 0}, {0: 6}, {0: 7}], index=[5, 1, 43, 23])
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            pd.DataFrame([{0: 5}, {0: 8}, {0: 2}], index=[19, 77, 4])
        )

    def test_split_without_test_with_pandas_and_counts_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=5,
            test_size=0,
            validation_size=5,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            pd.DataFrame([{0: 0}, {0: 1}, {0: 2}, {0: 3}, {0: 4}], index=[1, 3, 4, 5, 8])
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            pd.DataFrame([{0: 5}, {0: 6}, {0: 7}, {0: 8}, {0: 9}], index=[19, 43, 23, 77, 29])
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_split_without_test_with_pandas_and_counts_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=5,
            test_size=0,
            validation_size=5,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            pd.DataFrame([{0: 7}, {0: 1}, {0: 0}, {0: 6}, {0: 3}], index=[23, 3, 1, 43, 5])
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            pd.DataFrame([{0: 5}, {0: 8}, {0: 2}, {0: 9}, {0: 4}], index=[19, 77, 4, 29, 8])
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_without_test_or_validation_with_pandas_and_counts_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=10,
            test_size=0,
            validation_size=0,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            pd.DataFrame(range(10), index=index)
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_without_test_or_validation_with_pandas_and_counts_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=10,
            test_size=0,
            validation_size=0,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            pd.DataFrame(range(10), index=index)
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_without_train_or_validation_with_pandas_and_counts_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=0,
            test_size=10,
            validation_size=0,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            pd.DataFrame(range(10), index=index)
        )

    def test_without_train_or_validation_with_pandas_and_counts_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=0,
            test_size=10,
            validation_size=0,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            pd.DataFrame(range(10), index=index)
        )

    def test_defaults_without_train_or_validation_with_pandas_and_counts_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=0,
            test_size=None,
            validation_size=None,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            pd.DataFrame(range(10), index=index)
        )

    def test_defaults_without_train_or_validation_with_pandas_and_counts_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=0,
            test_size=None,
            validation_size=None,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            pd.DataFrame(range(10), index=index)
        )

    def test_without_train_or_test_with_pandas_and_counts_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=0,
            test_size=0,
            validation_size=10,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            pd.DataFrame(range(10), index=index)
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_without_train_or_test_with_pandas_and_counts_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=0,
            test_size=0,
            validation_size=10,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            pd.DataFrame(range(10), index=index)
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_defaults_without_train_or_test_with_pandas_and_counts_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=0,
            test_size=0,
            validation_size=None,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            pd.DataFrame(range(10), index=index)
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_defaults_without_train_or_test_with_pandas_and_counts_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=0,
            test_size=0,
            validation_size=None,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            pd.DataFrame(range(10), index=index)
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    '''
    Rates + Pandas
    '''

    def test_all_splits_with_pandas_and_rates_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=0.3,
            test_size=0.3,
            validation_size=0.4,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            pd.DataFrame([{0: 0}, {0: 1}, {0: 2}], index=[1, 3, 4])
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            pd.DataFrame([{0: 3}, {0: 4}, {0: 5}, {0: 6}], index=[5, 8, 19, 43])
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            pd.DataFrame([{0: 7}, {0: 8}, {0: 9}], index=[23, 77, 29])
        )

    def test_all_splits_with_pandas_and_rates_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=0.3,
            test_size=0.3,
            validation_size=0.4,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            pd.DataFrame([{0: 4}, {0: 9}, {0: 1}], index=[8, 29, 3])
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            pd.DataFrame([{0: 3}, {0: 0}, {0: 6}, {0: 7}], index=[5, 1, 43, 23])
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            pd.DataFrame([{0: 5}, {0: 8}, {0: 2}], index=[19, 77, 4])
        )

    def test_split_without_test_with_pandas_and_rates_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=0.5,
            test_size=0.0,
            validation_size=0.5,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            pd.DataFrame([{0: 0}, {0: 1}, {0: 2}, {0: 3}, {0: 4}], index=[1, 3, 4, 5, 8])
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            pd.DataFrame([{0: 5}, {0: 6}, {0: 7}, {0: 8}, {0: 9}], index=[19, 43, 23, 77, 29])
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_split_without_test_with_pandas_and_rates_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=0.5,
            test_size=0.0,
            validation_size=0.5,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            pd.DataFrame([{0: 7}, {0: 1}, {0: 0}, {0: 6}, {0: 3}], index=[23, 3, 1, 43, 5])
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            pd.DataFrame([{0: 5}, {0: 8}, {0: 2}, {0: 9}, {0: 4}], index=[19, 77, 4, 29, 8])
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_without_test_or_validation_with_pandas_and_rates_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=1.0,
            test_size=0.0,
            validation_size=0.0,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            pd.DataFrame(range(10), index=index)
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_without_test_or_validation_with_pandas_and_rates_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=1.0,
            test_size=0.0,
            validation_size=0.0,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            pd.DataFrame(range(10), index=index)
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_without_train_or_validation_with_pandas_and_rates_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=0.0,
            test_size=1.0,
            validation_size=0.0,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            pd.DataFrame(range(10), index=index)
        )

    def test_without_train_or_validation_with_pandas_and_rates_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=0.0,
            test_size=1.0,
            validation_size=0.0,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            pd.DataFrame(range(10), index=index)
        )

    def test_defaults_without_train_or_validation_with_pandas_and_rates_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=0.0,
            test_size=None,
            validation_size=None,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            pd.DataFrame(range(10), index=index)
        )

    def test_defaults_without_train_or_validation_with_pandas_and_rates_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=0.0,
            test_size=None,
            validation_size=None,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            pd.DataFrame(range(10), index=index)
        )

    def test_without_train_or_test_with_pandas_and_rates_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=0.0,
            test_size=0.0,
            validation_size=1.0,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            pd.DataFrame(range(10), index=index)
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_without_train_or_test_with_pandas_and_rates_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=0.0,
            test_size=0.0,
            validation_size=1.0,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            pd.DataFrame(range(10), index=index)
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_defaults_without_train_or_test_with_pandas_and_rates_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=0.0,
            test_size=0.0,
            validation_size=None,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            pd.DataFrame(range(10), index=index)
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_defaults_without_train_or_test_with_pandas_and_rates_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = pd.DataFrame(range(10), index=index)
        mixin = self.TestRandomSplitMixin(
            train_size=0.0,
            test_size=0.0,
            validation_size=None,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            pd.DataFrame(range(10), index=index)
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    '''
    Dask - dask index is sorted so selection is offset from
    mixed pandas index (necessary to enable index based selection)
    '''

    def test_default_all_splits_with_dask_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=None,
            test_size=None,
            validation_size=None,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=2, sort=True)
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_default_all_splits_with_dask_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=None,
            test_size=None,
            validation_size=None,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=2, sort=True)
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    '''
    Counts + Dask
    '''

    def test_all_splits_with_dask_and_counts_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=3,
            test_size=3,
            validation_size=4,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            dd.from_pandas(pd.DataFrame([{0: 0}, {0: 1}, {0: 2}], index=[1, 3, 4]), npartitions=2, sort=False)
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            dd.from_pandas(pd.DataFrame([{0: 3}, {0: 4}, {0: 5}, {0: 7}], index=[5, 8, 19, 23]), npartitions=2, sort=False)
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            dd.from_pandas(pd.DataFrame([{0: 9}, {0: 6}, {0: 8}], index=[29, 43, 77]), npartitions=2, sort=False)
        )

    def test_all_splits_with_dask_and_counts_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=3,
            test_size=3,
            validation_size=4,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            dd.from_pandas(pd.DataFrame([{0: 1}, {0: 4}, {0: 8}], index=[3, 8, 77]), npartitions=2, sort=False)
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            dd.from_pandas(pd.DataFrame([{0: 0}, {0: 3}, {0: 7}, {0: 9}], index=[1, 5, 23, 29]), npartitions=2, sort=False)
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            dd.from_pandas(pd.DataFrame([{0: 2}, {0: 5}, {0: 6}], index=[4, 19, 43]), npartitions=2, sort=False)
        )

    def test_split_without_test_with_dask_and_counts_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=5,
            test_size=0,
            validation_size=5,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            dd.from_pandas(pd.DataFrame([{0: 0}, {0: 1}, {0: 2}, {0: 3}, {0: 4}], index=[1, 3, 4, 5, 8]), npartitions=2, sort=False)
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            dd.from_pandas(pd.DataFrame([{0: 5}, {0: 7}, {0: 9}, {0: 6}, {0: 8}], index=[19, 23, 29, 43, 77]), npartitions=2, sort=False)
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_split_without_test_with_dask_and_counts_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=5,
            test_size=0,
            validation_size=5,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            dd.from_pandas(pd.DataFrame([{0: 1}, {0: 0}, {0: 3}, {0: 9}, {0: 7}], index=[3, 1, 5, 29, 23]), npartitions=2, sort=False)
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            dd.from_pandas(pd.DataFrame([{0: 2}, {0: 5}, {0: 4}, {0: 6}, {0: 8}], index=[4, 19, 8, 43, 77]), npartitions=2, sort=False)
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_without_test_or_validation_with_dask_and_counts_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=10,
            test_size=0,
            validation_size=0,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=2, sort=True)
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_without_test_or_validation_with_dask_and_counts_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=10,
            test_size=0,
            validation_size=0,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=2, sort=True)
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_without_train_or_validation_with_dask_and_counts_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=0,
            test_size=10,
            validation_size=0,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=2, sort=True)
        )

    def test_without_train_or_validation_with_dask_and_counts_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=0,
            test_size=10,
            validation_size=0,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=2, sort=True)
        )

    def test_defaults_without_train_or_validation_with_dask_and_counts_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=0,
            test_size=None,
            validation_size=None,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=2, sort=True)
        )

    def test_defaults_without_train_or_validation_with_dask_and_counts_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=0,
            test_size=None,
            validation_size=None,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=2, sort=True)
        )

    def test_without_train_or_test_with_dask_and_counts_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=0,
            test_size=0,
            validation_size=10,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=2, sort=True)
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_without_train_or_test_with_dask_and_counts_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=0,
            test_size=0,
            validation_size=10,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=2, sort=True)
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_defaults_without_train_or_test_with_dask_and_counts_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=0,
            test_size=0,
            validation_size=None,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=2, sort=True)
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_defaults_without_train_or_test_with_dask_and_counts_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=0,
            test_size=0,
            validation_size=None,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=2, sort=True)
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    '''
    Rates + Dask
    '''

    def test_all_splits_with_dask_and_rates_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=0.3,
            test_size=0.3,
            validation_size=0.4,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            dd.from_pandas(pd.DataFrame([{0: 0}, {0: 1}, {0: 2}], index=[1, 3, 4]), npartitions=2, sort=False)
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            dd.from_pandas(pd.DataFrame([{0: 3}, {0: 4}, {0: 5}, {0: 7}], index=[5, 8, 19, 23]), npartitions=2, sort=False)
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            dd.from_pandas(pd.DataFrame([{0: 9}, {0: 6}, {0: 8}], index=[29, 43, 77]), npartitions=2, sort=False)
        )

    def test_all_splits_with_dask_and_rates_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=0.3,
            test_size=0.3,
            validation_size=0.4,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            dd.from_pandas(pd.DataFrame([{0: 1}, {0: 4}, {0: 8}], index=[3, 8, 77]), npartitions=2, sort=False)
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            dd.from_pandas(pd.DataFrame([{0: 0}, {0: 3}, {0: 7}, {0: 9}], index=[1, 5, 23, 29]), npartitions=2, sort=False)
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            dd.from_pandas(pd.DataFrame([{0: 2}, {0: 5}, {0: 6}], index=[4, 19, 43]), npartitions=2, sort=False)
        )

    def test_split_without_test_with_dask_and_rates_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=0.5,
            test_size=0.0,
            validation_size=0.5,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            dd.from_pandas(pd.DataFrame([{0: 0}, {0: 1}, {0: 2}, {0: 3}, {0: 4}], index=[1, 3, 4, 5, 8]), npartitions=2, sort=False)
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            dd.from_pandas(pd.DataFrame([{0: 5}, {0: 7}, {0: 9}, {0: 6}, {0: 8}], index=[19, 23, 29, 43, 77]), npartitions=2, sort=False)
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_split_without_test_with_dask_and_rates_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=0.5,
            test_size=0.0,
            validation_size=0.5,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            dd.from_pandas(pd.DataFrame([{0: 1}, {0: 0}, {0: 3}, {0: 9}, {0: 7}], index=[3, 1, 5, 29, 23]), npartitions=2, sort=False)
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            dd.from_pandas(pd.DataFrame([{0: 2}, {0: 5}, {0: 4}, {0: 6}, {0: 8}], index=[4, 19, 8, 43, 77]), npartitions=2, sort=False)
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_without_test_or_validation_with_dask_and_rates_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=1.0,
            test_size=0.0,
            validation_size=0.0,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=2, sort=True)
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_without_test_or_validation_with_dask_and_rates_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=1.0,
            test_size=0.0,
            validation_size=0.0,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=2, sort=True)
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_without_train_or_validation_with_dask_and_rates_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=0.0,
            test_size=1.0,
            validation_size=0.0,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=2, sort=True)
        )

    def test_without_train_or_validation_with_dask_and_rates_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=0.0,
            test_size=1.0,
            validation_size=0.0,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=2, sort=True)
        )

    def test_defaults_without_train_or_validation_with_dask_and_rates_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=0.0,
            test_size=None,
            validation_size=None,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=2, sort=True)
        )

    def test_defaults_without_train_or_validation_with_dask_and_rates_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=0.0,
            test_size=None,
            validation_size=None,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=2, sort=True)
        )

    def test_without_train_or_test_with_dask_and_rates_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=0.0,
            test_size=0.0,
            validation_size=1.0,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=2, sort=True)
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_without_train_or_test_with_dask_and_rates_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=0.0,
            test_size=0.0,
            validation_size=1.0,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=2, sort=True)
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_defaults_without_train_or_test_with_dask_and_rates_no_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=0.0,
            test_size=0.0,
            validation_size=None,
            random_state=23,
            shuffle=False
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=2, sort=True)
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )

    def test_defaults_without_train_or_test_with_dask_and_rates_with_shuffle(self):
        index = [1, 3, 4, 5, 8, 19, 43, 23, 77, 29]
        data = dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=5, sort=True)
        mixin = self.TestRandomSplitMixin(
            train_size=0.0,
            test_size=0.0,
            validation_size=None,
            random_state=23,
            shuffle=True
        )
        mixin.dataset = self.MockDataset(Split(X=data))
        mixin.split_dataset()

        assert_data_container_equal(
            mixin._dataset_splits['TRAIN'].X,
            None
        )
        assert_data_container_equal(
            mixin._dataset_splits['VALIDATION'].X,
            dd.from_pandas(pd.DataFrame(range(10), index=index), npartitions=2, sort=True)
        )
        assert_data_container_equal(
            mixin._dataset_splits['TEST'].X,
            None
        )


class ExplicitSplitMixinTests(unittest.TestCase):
    pass


class ChronologicalSplitMixinTests(unittest.TestCase):
    pass


class KFoldSplitMixinTests(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
