'''
Projected Split related tests
'''

__author__ = 'Elisha Yadgaran'


import unittest
import pandas as pd
import numpy as np

from unittest.mock import MagicMock, patch, PropertyMock
from abc import ABCMeta, abstractmethod

from simpleml.datasets.base_dataset import Dataset
from simpleml.datasets.dataset_splits import Split
from simpleml.pipelines.projected_splits import ProjectedDatasetSplit, IdentityProjectedDatasetSplit, IndexBasedProjectedDatasetSplit


class AbstractProjectedDatasetSplitTests(unittest.TestCase):

    def test_projection_abstractness(self):
        '''
        Ensure logic definition is required for classes
        '''
        with self.assertRaises(TypeError):
            ProjectedDatasetSplit(dataset=None, split=None)


class ProjectedDatasetSplitTests(metaclass=ABCMeta):
    @property
    @abstractmethod
    def test_cls(self):
        pass

    def mock_cls(self, dataset=None, split=None, **kwargs):
        return self.test_cls(dataset=dataset, split=split, **kwargs)

    def test_getting_dataset_split(self):
        '''
        Test that the method passes through as expected
        Uses a mock dataset so not particularly meaningful behavior
        '''
        mock_dataset = MagicMock()
        projected_split = self.mock_cls(dataset=mock_dataset, split='asdf')
        split = projected_split.dataset_split

        self.assertTrue(isinstance(split, MagicMock))
        mock_dataset.get_split.assert_called_with(split='asdf')

    @abstractmethod
    def test_projection_logic(self):
        '''
        Test that the projection works as expected
        '''

    def test_passthrough_behavior(self):
        '''
        Test the split is proxied for references
        '''
        prop_mock = PropertyMock()
        with patch.object(self.test_cls, 'projected_split', new_callable=prop_mock) as mocked_method:
            projected_split = self.mock_cls()

            for param in [
                'X', 'y', 'other', 'blah', 'blue'
            ]:
                # mocks create attributes on access. assert it doesnt exist and
                # gets created when called on the projected split
                self.assertFalse(param in dir(mocked_method))
                getattr(projected_split, param)(param)
                self.assertTrue(hasattr(mocked_method, param))
                getattr(mocked_method, param).assert_called_once_with(param)

    def test_itemgetter_behavior(self):
        '''
        Test the split is proxied for references
        '''
        mock_split = Split(X='abc', y='def', other='xyz')
        prop_mock = PropertyMock(return_value=mock_split)
        with patch.object(self.test_cls, 'projected_split', new_callable=prop_mock) as mocked_method:
            projected_split = self.mock_cls()

            for section in mock_split:
                self.assertEqual(mock_split[section], projected_split[section])

            # **behavior
            self.assertEqual({**mock_split}, {**projected_split})
            self.assertEqual({'X': 'abc', 'y': 'def', 'other': 'xyz'}, {**projected_split})


class IdentityProjectedDatasetSplitTests(unittest.TestCase, ProjectedDatasetSplitTests):
    @property
    def test_cls(self):
        return IdentityProjectedDatasetSplit

    def test_projection_logic(self):
        '''
        Test that the projection works as expected
        '''
        mock_dataset = MagicMock()
        mock_dataset.get_split.return_value = Split(X='a', y='b')
        projected_split = self.mock_cls(dataset=mock_dataset, split='ddefg')
        split = projected_split.dataset_split
        explicit_output = projected_split.apply_projection(split)
        implicit_output = projected_split.projected_split

        self.assertEqual(split, explicit_output)
        self.assertEqual(split, implicit_output)
        self.assertEqual(implicit_output, explicit_output)


class IndexProjectedDatasetSplitTests(unittest.TestCase, ProjectedDatasetSplitTests):
    @property
    def test_cls(self):
        return IndexBasedProjectedDatasetSplit

    def mock_cls(self, dataset=None, split=None, indices=None, **kwargs):
        return self.test_cls(dataset=dataset, split=split, indices=indices, **kwargs)

    def test_projection_logic(self):
        '''
        Test that the projection works as expected for pandas objects
        '''
        mock_dataset = MagicMock()
        mock_dataset.get_split.return_value = Split(X=pd.DataFrame(range(100)), y=pd.Series(range(100)))
        projected_split = self.mock_cls(dataset=mock_dataset, split='ddefg', indices=range(10, 30))
        expected_split = Split(X=pd.DataFrame(range(10, 30), index=range(10, 30)), y=pd.Series(range(10, 30), index=range(10, 30)))
        split = projected_split.dataset_split
        explicit_output = projected_split.apply_projection(split)
        implicit_output = projected_split.projected_split

        def pandas_split_comparison(a, b):
            self.assertEqual(a.keys(), b.keys())
            for k, v in a.items():
                if isinstance(v, pd.DataFrame):
                    pd.testing.assert_frame_equal(v, b[k])
                else:
                    pd.testing.assert_series_equal(v, b[k])

        with self.assertRaises(AssertionError):
            pandas_split_comparison(split, explicit_output)

        with self.assertRaises(AssertionError):
            pandas_split_comparison(split, implicit_output)

        pandas_split_comparison(expected_split, explicit_output)
        pandas_split_comparison(expected_split, implicit_output)
        pandas_split_comparison(implicit_output, explicit_output)

    def test_numpy_projection_logic(self):
        '''
        Test that the projection works as expected for numpy objects
        '''
        mock_dataset = MagicMock()
        mock_dataset.get_split.return_value = Split(X=np.ones((100, 10)) * np.array(range(100)).reshape(-1, 1), y=np.array(range(100)))
        projected_split = self.mock_cls(dataset=mock_dataset, split='ddefg', indices=range(10, 30))
        expected_split = Split(X=np.ones((20, 10)) * np.array(range(10, 30)).reshape(-1, 1), y=np.array(range(10, 30)))
        split = projected_split.dataset_split
        explicit_output = projected_split.apply_projection(split)
        implicit_output = projected_split.projected_split

        def numpy_split_comparison(a, b):
            self.assertEqual(a.keys(), b.keys())
            for k, v in a.items():
                np.testing.assert_equal(v, b[k])

        with self.assertRaises(AssertionError):
            numpy_split_comparison(split, explicit_output)

        with self.assertRaises(AssertionError):
            numpy_split_comparison(split, implicit_output)

        numpy_split_comparison(expected_split, explicit_output)
        numpy_split_comparison(expected_split, implicit_output)
        numpy_split_comparison(implicit_output, explicit_output)


if __name__ == '__main__':
    unittest.main(verbosity=2)
