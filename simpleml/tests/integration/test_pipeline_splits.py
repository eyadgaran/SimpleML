'''
Integration tests for pipeline split behavior
'''

__author__ = 'Elisha Yadgaran'


import unittest
import pandas as pd

from abc import ABC, abstractmethod
from simpleml.tests.utils import assert_split_equal
from simpleml.datasets import SingleLabelPandasDataset, MultiLabelPandasDataset
from simpleml.datasets.dataset_splits import Split
from simpleml.pipelines import NoSplitPipeline, ExplicitSplitPipeline, RandomSplitPipeline
from simpleml.pipelines.projected_splits import ProjectedDatasetSplit


class _PipelineSplitTests(ABC):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''

    def setUp(self):
        self.dataset = self.dataset_cls(**self.dataset_params)
        self.dataset.dataframe = self.build_dataset()
        self.pipeline = self.pipeline_cls(**self.pipeline_params)
        self.pipeline.add_dataset(self.dataset)

    @property
    def dataset_params(self):
        return {}

    @property
    def pipeline_params(self):
        return {}

    @abstractmethod
    def expected_split_contents(self):
        pass

    @abstractmethod
    def build_dataset(self):
        pass

    @abstractmethod
    def example_split_name(self):
        pass

    def test_getting_splits_with_mutation(self):
        '''
        Mutate split and re-retrieve
        No split behavior passes all the data for any split
        '''
        split = self.pipeline.get_dataset_split(split=self.example_split_name())
        projected_split = split.projected_split
        self.assertTrue(isinstance(split, ProjectedDatasetSplit))
        self.assertTrue(isinstance(projected_split, Split))

        assert_split_equal(projected_split, self.expected_split_contents())
        assert_split_equal(split, self.expected_split_contents())

        # mutate
        projected_split['X'] = None
        with self.assertRaises(AssertionError):
            assert_split_equal(projected_split, self.expected_split_contents())

        # assert equality
        new_split = self.pipeline.get_dataset_split(split=self.example_split_name())
        assert_split_equal(split, self.expected_split_contents())
        assert_split_equal(new_split, self.expected_split_contents())


class NoSplitPipelineSingleLabelPandasDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = SingleLabelPandasDataset
    pipeline_cls = NoSplitPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c'], 'other_named_split_sections': {'other': ['e']}}

    def example_split_name(self):
        return None

    def expected_split_contents(self):
        return Split(
            X=pd.DataFrame([
                {'a': 1, 'b': 2},
                {'a': 2, 'b': 3},
                {'a': 3, 'b': 4},
                {'a': 4, 'b': 5},
                {'a': 5, 'b': 6},
                {'a': 6, 'b': 7}]),
            y=pd.Series([3, 4, 5, 6, 7, 8], name='c'),
            other=pd.Series([5, 6, 7, 8, 9, 10], name='e'))

    def build_dataset(self):
        return self.dataset_cls.concatenate_dataframes(
            dataframes=[
                pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'e': 5}, {'a': 2, 'b': 3, 'c': 4, 'e': 6}]),
                pd.DataFrame([{'a': 3, 'b': 4, 'c': 5, 'e': 7}, {'a': 4, 'b': 5, 'c': 6, 'e': 8}]),
                pd.DataFrame([{'a': 5, 'b': 6, 'c': 7, 'e': 9}, {'a': 6, 'b': 7, 'c': 8, 'e': 10}]),
            ],
            split_names=['first', 'second', 'third']
        )


class ExplicitSplitPipelineSingleLabelPandasDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = SingleLabelPandasDataset
    pipeline_cls = ExplicitSplitPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c'], 'other_named_split_sections': {'other': ['e']}}

    def example_split_name(self):
        return 'first'

    def expected_split_contents(self):
        return Split(
            X=pd.DataFrame([
                {'a': 1, 'b': 2},
                {'a': 2, 'b': 3}]),
            y=pd.Series([3, 4], name='c'),
            other=pd.Series([5, 6], name='e'))

    def build_dataset(self):
        return self.dataset_cls.concatenate_dataframes(
            dataframes=[
                pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'e': 5}, {'a': 2, 'b': 3, 'c': 4, 'e': 6}]),
                pd.DataFrame([{'a': 3, 'b': 4, 'c': 5, 'e': 7}, {'a': 4, 'b': 5, 'c': 6, 'e': 8}]),
                pd.DataFrame([{'a': 5, 'b': 6, 'c': 7, 'e': 9}, {'a': 6, 'b': 7, 'c': 8, 'e': 10}]),
            ],
            split_names=['first', 'second', 'third']
        )


class RandomSplitPipelineSingleLabelPandasDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = SingleLabelPandasDataset
    pipeline_cls = RandomSplitPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c'], 'other_named_split_sections': {'other': ['e']}}

    @property
    def pipeline_params(self):
        return {'train_size': 0.5, 'random_state': 10}

    def example_split_name(self):
        return 'TRAIN'

    def expected_split_contents(self):
        return Split(
            X=pd.DataFrame([
                {'a': 4, 'b': 5},
                {'a': 5, 'b': 6},
                {'a': 2, 'b': 3}],
                index=[3, 4, 1]),
            y=pd.Series([6, 7, 4], index=[3, 4, 1], name='c'),
            other=pd.Series([8, 9, 6], index=[3, 4, 1], name='e'))

    def build_dataset(self):
        return self.dataset_cls.concatenate_dataframes(
            dataframes=[
                pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'e': 5}, {'a': 2, 'b': 3, 'c': 4, 'e': 6}]),
                pd.DataFrame([{'a': 3, 'b': 4, 'c': 5, 'e': 7}, {'a': 4, 'b': 5, 'c': 6, 'e': 8}]),
                pd.DataFrame([{'a': 5, 'b': 6, 'c': 7, 'e': 9}, {'a': 6, 'b': 7, 'c': 8, 'e': 10}]),
            ],
            split_names=['first', 'second', 'third']
        )


class NoSplitPipelineMultiLabelPandasDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = MultiLabelPandasDataset
    pipeline_cls = NoSplitPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c', 'd'], 'other_named_split_sections': {'other': ['e']}}

    def example_split_name(self):
        return None

    def expected_split_contents(self):
        return Split(
            X=pd.DataFrame([
                {'a': 1, 'b': 2},
                {'a': 2, 'b': 3},
                {'a': 3, 'b': 4},
                {'a': 4, 'b': 5},
                {'a': 5, 'b': 6},
                {'a': 6, 'b': 7}]),
            y=pd.DataFrame([
                {'c': 3, 'd': 4},
                {'c': 4, 'd': 5},
                {'c': 5, 'd': 6},
                {'c': 6, 'd': 7},
                {'c': 7, 'd': 8},
                {'c': 8, 'd': 9}]),
            other=pd.DataFrame([
                {'e': 5},
                {'e': 6},
                {'e': 7},
                {'e': 8},
                {'e': 9},
                {'e': 10}])
        )

    def build_dataset(self):
        return self.dataset_cls.concatenate_dataframes(
            dataframes=[
                pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}, {'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6}]),
                pd.DataFrame([{'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7}, {'a': 4, 'b': 5, 'c': 6, 'd': 7, 'e': 8}]),
                pd.DataFrame([{'a': 5, 'b': 6, 'c': 7, 'd': 8, 'e': 9}, {'a': 6, 'b': 7, 'c': 8, 'd': 9, 'e': 10}]),
            ],
            split_names=['first', 'second', 'third']
        )


class ExplicitSplitPipelineMultiLabelPandasDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = MultiLabelPandasDataset
    pipeline_cls = ExplicitSplitPipeline

    @ property
    def dataset_params(self):
        return {'label_columns': ['c', 'd'], 'other_named_split_sections': {'other': ['e']}}

    def example_split_name(self):
        return 'first'

    def expected_split_contents(self):
        return Split(
            X=pd.DataFrame([
                {'a': 1, 'b': 2},
                {'a': 2, 'b': 3}]),
            y=pd.DataFrame([
                {'c': 3, 'd': 4},
                {'c': 4, 'd': 5}]),
            other=pd.DataFrame([
                {'e': 5},
                {'e': 6}])
        )

    def build_dataset(self):
        return self.dataset_cls.concatenate_dataframes(
            dataframes=[
                pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}, {'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6}]),
                pd.DataFrame([{'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7}, {'a': 4, 'b': 5, 'c': 6, 'd': 7, 'e': 8}]),
                pd.DataFrame([{'a': 5, 'b': 6, 'c': 7, 'd': 8, 'e': 9}, {'a': 6, 'b': 7, 'c': 8, 'd': 9, 'e': 10}]),
            ],
            split_names=['first', 'second', 'third']
        )


class RandomSplitPipelineMultiLabelPandasDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = MultiLabelPandasDataset
    pipeline_cls = RandomSplitPipeline

    @ property
    def dataset_params(self):
        return {'label_columns': ['c', 'd'], 'other_named_split_sections': {'other': ['e']}}

    @ property
    def pipeline_params(self):
        return {'train_size': 0.5, 'random_state': 10}

    def example_split_name(self):
        return 'TRAIN'

    def expected_split_contents(self):
        return Split(
            X=pd.DataFrame([
                {'a': 4, 'b': 5},
                {'a': 5, 'b': 6},
                {'a': 2, 'b': 3}],
                index=[3, 4, 1]),
            y=pd.DataFrame([
                {'c': 6, 'd': 7},
                {'c': 7, 'd': 8},
                {'c': 4, 'd': 5}],
                index=[3, 4, 1]),
            other=pd.DataFrame([
                {'e': 8},
                {'e': 9},
                {'e': 6}],
                index=[3, 4, 1])
        )

    def build_dataset(self):
        return self.dataset_cls.concatenate_dataframes(
            dataframes=[
                pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}, {'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6}]),
                pd.DataFrame([{'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7}, {'a': 4, 'b': 5, 'c': 6, 'd': 7, 'e': 8}]),
                pd.DataFrame([{'a': 5, 'b': 6, 'c': 7, 'd': 8, 'e': 9}, {'a': 6, 'b': 7, 'c': 8, 'd': 9, 'e': 10}]),
            ],
            split_names=['first', 'second', 'third']
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)
