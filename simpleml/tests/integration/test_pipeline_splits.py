'''
Integration tests for pipeline split behavior
'''

__author__ = 'Elisha Yadgaran'


import unittest
from abc import ABC, abstractmethod

import pandas as pd

from simpleml.datasets.dask.base import BaseDaskDataset
from simpleml.datasets.dataset_splits import Split
from simpleml.datasets.pandas.base import BasePandasDataset
from simpleml.imports import dd
from simpleml.pipelines.ordered_dict import (
    ExplicitSplitOrderedDictPipeline,
    OrderedDictPipeline,
    RandomSplitOrderedDictPipeline,
)
from simpleml.pipelines.projected_splits import ProjectedDatasetSplit
from simpleml.pipelines.sklearn import (
    ExplicitSplitSklearnPipeline,
    RandomSplitSklearnPipeline,
    SklearnPipeline,
)
from simpleml.tests.utils import assert_split_equal
from simpleml.transformers import Transformer


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
        # Assertion error for pandas, Attribute error for Dask (tries to call compute on NoneType)
        with self.assertRaises((AssertionError, AttributeError)):
            assert_split_equal(projected_split, self.expected_split_contents())

        # assert equality
        new_split = self.pipeline.get_dataset_split(split=self.example_split_name())
        assert_split_equal(split, self.expected_split_contents())
        assert_split_equal(new_split, self.expected_split_contents())


'''
Sklearn + Pandas
'''


class SklearnPipelineSingleLabelPandasDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = BasePandasDataset
    pipeline_cls = SklearnPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c'], 'other_named_split_sections': {'other': ['e']}, 'squeeze_return': True}

    @property
    def pipeline_params(self):
        '''
        Sklearn cannot implement a pipeline without transformers
        '''
        return {'transformers': [('step', Transformer())]}

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


class ExplicitSplitSklearnPipelineSingleLabelPandasDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = BasePandasDataset
    pipeline_cls = ExplicitSplitSklearnPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c'], 'other_named_split_sections': {'other': ['e']}, 'squeeze_return': True}

    @property
    def pipeline_params(self):
        '''
        Sklearn cannot implement a pipeline without transformers
        '''
        return {'transformers': [('step', Transformer())]}

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


class RandomSplitSklearnPipelineSingleLabelPandasDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = BasePandasDataset
    pipeline_cls = RandomSplitSklearnPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c'], 'other_named_split_sections': {'other': ['e']}, 'squeeze_return': True}

    @property
    def pipeline_params(self):
        '''
        Sklearn cannot implement a pipeline without transformers
        '''
        return {'train_size': 0.5, 'random_state': 10, 'transformers': [('step', Transformer())]}

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


class SklearnPipelineMultiLabelPandasDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = BasePandasDataset
    pipeline_cls = SklearnPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c', 'd'], 'other_named_split_sections': {'other': ['e']}, 'squeeze_return': False}

    @property
    def pipeline_params(self):
        '''
        Sklearn cannot implement a pipeline without transformers
        '''
        return {'transformers': [('step', Transformer())]}

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


class ExplicitSplitSklearnPipelineMultiLabelPandasDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = BasePandasDataset
    pipeline_cls = ExplicitSplitSklearnPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c', 'd'], 'other_named_split_sections': {'other': ['e']}, 'squeeze_return': False}

    @property
    def pipeline_params(self):
        '''
        Sklearn cannot implement a pipeline without transformers
        '''
        return {'transformers': [('step', Transformer())]}

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


class RandomSplitSklearnPipelineMultiLabelPandasDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = BasePandasDataset
    pipeline_cls = RandomSplitSklearnPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c', 'd'], 'other_named_split_sections': {'other': ['e']}, 'squeeze_return': False}

    @property
    def pipeline_params(self):
        '''
        Sklearn cannot implement a pipeline without transformers
        '''
        return {'train_size': 0.5, 'random_state': 10, 'transformers': [('step', Transformer())]}

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


'''
OrderedDict + Pandas
'''


class OrderedDictPipelineSingleLabelPandasDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = BasePandasDataset
    pipeline_cls = OrderedDictPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c'], 'other_named_split_sections': {'other': ['e']}, 'squeeze_return': True}

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


class ExplicitSplitOrderedDictPipelineSingleLabelPandasDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = BasePandasDataset
    pipeline_cls = ExplicitSplitOrderedDictPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c'], 'other_named_split_sections': {'other': ['e']}, 'squeeze_return': True}

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


class RandomSplitOrderedDictPipelineSingleLabelPandasDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = BasePandasDataset
    pipeline_cls = RandomSplitOrderedDictPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c'], 'other_named_split_sections': {'other': ['e']}, 'squeeze_return': True}

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


class OrderedDictPipelineMultiLabelPandasDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = BasePandasDataset
    pipeline_cls = OrderedDictPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c', 'd'], 'other_named_split_sections': {'other': ['e']}, 'squeeze_return': False}

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


class ExplicitSplitOrderedDictPipelineMultiLabelPandasDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = BasePandasDataset
    pipeline_cls = ExplicitSplitOrderedDictPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c', 'd'], 'other_named_split_sections': {'other': ['e']}, 'squeeze_return': False}

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


class RandomSplitOrderedDictPipelineMultiLabelPandasDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = BasePandasDataset
    pipeline_cls = RandomSplitOrderedDictPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c', 'd'], 'other_named_split_sections': {'other': ['e']}, 'squeeze_return': False}

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


'''
Sklearn + Dask
'''


class SklearnPipelineSingleLabelDaskDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = BaseDaskDataset
    pipeline_cls = SklearnPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c'], 'other_named_split_sections': {'other': ['e']}, 'squeeze_return': True}

    @property
    def pipeline_params(self):
        '''
        Sklearn cannot implement a pipeline without transformers
        '''
        return {'transformers': [('step', Transformer())]}

    def example_split_name(self):
        return None

    def expected_split_contents(self):
        return Split(
            X=dd.from_pandas(pd.DataFrame([
                {'a': 1, 'b': 2},
                {'a': 2, 'b': 3},
                {'a': 3, 'b': 4},
                {'a': 4, 'b': 5},
                {'a': 5, 'b': 6},
                {'a': 6, 'b': 7}]),
                npartitions=5, sort=True),
            y=dd.from_pandas(pd.Series([3, 4, 5, 6, 7, 8], name='c'),
                             npartitions=5, sort=True),
            other=dd.from_pandas(pd.Series([5, 6, 7, 8, 9, 10], name='e'),
                                 npartitions=5, sort=True))

    def build_dataset(self):
        # use pandas concat to handle index deduplication
        return dd.from_pandas(BasePandasDataset.concatenate_dataframes(
            dataframes=[
                pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'e': 5}, {'a': 2, 'b': 3, 'c': 4, 'e': 6}]),
                pd.DataFrame([{'a': 3, 'b': 4, 'c': 5, 'e': 7}, {'a': 4, 'b': 5, 'c': 6, 'e': 8}]),
                pd.DataFrame([{'a': 5, 'b': 6, 'c': 7, 'e': 9}, {'a': 6, 'b': 7, 'c': 8, 'e': 10}]),
            ],
            split_names=['first', 'second', 'third']
        ), npartitions=5, sort=True)


class ExplicitSplitSklearnPipelineSingleLabelDaskDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = BaseDaskDataset
    pipeline_cls = ExplicitSplitSklearnPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c'], 'other_named_split_sections': {'other': ['e']}, 'squeeze_return': True}

    @property
    def pipeline_params(self):
        '''
        Sklearn cannot implement a pipeline without transformers
        '''
        return {'transformers': [('step', Transformer())]}

    def example_split_name(self):
        return 'first'

    def expected_split_contents(self):
        return Split(
            X=dd.from_pandas(pd.DataFrame([
                {'a': 1, 'b': 2},
                {'a': 2, 'b': 3}]),
                npartitions=5, sort=True),
            y=dd.from_pandas(pd.Series([3, 4], name='c'),
                             npartitions=5, sort=True),
            other=dd.from_pandas(pd.Series([5, 6], name='e'),
                                 npartitions=5, sort=True))

    def build_dataset(self):
        # use pandas concat to handle index deduplication
        return dd.from_pandas(BasePandasDataset.concatenate_dataframes(
            dataframes=[
                pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'e': 5}, {'a': 2, 'b': 3, 'c': 4, 'e': 6}]),
                pd.DataFrame([{'a': 3, 'b': 4, 'c': 5, 'e': 7}, {'a': 4, 'b': 5, 'c': 6, 'e': 8}]),
                pd.DataFrame([{'a': 5, 'b': 6, 'c': 7, 'e': 9}, {'a': 6, 'b': 7, 'c': 8, 'e': 10}]),
            ],
            split_names=['first', 'second', 'third']
        ), npartitions=5, sort=True)


class RandomSplitSklearnPipelineSingleLabelDaskDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = BaseDaskDataset
    pipeline_cls = RandomSplitSklearnPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c'], 'other_named_split_sections': {'other': ['e']}, 'squeeze_return': True}

    @property
    def pipeline_params(self):
        '''
        Sklearn cannot implement a pipeline without transformers
        '''
        return {'train_size': 0.5, 'random_state': 10, 'transformers': [('step', Transformer())]}

    def example_split_name(self):
        return 'TRAIN'

    def expected_split_contents(self):
        return Split(
            X=dd.from_pandas(pd.DataFrame([
                {'a': 4, 'b': 5},
                {'a': 5, 'b': 6},
                {'a': 2, 'b': 3}],
                index=[3, 4, 1]),
                npartitions=5, sort=True),
            y=dd.from_pandas(pd.Series([6, 7, 4], index=[3, 4, 1], name='c'),
                             npartitions=5, sort=True),
            other=dd.from_pandas(pd.Series([8, 9, 6], index=[3, 4, 1], name='e'),
                                 npartitions=5, sort=True))

    def build_dataset(self):
        # use pandas concat to handle index deduplication
        return dd.from_pandas(BasePandasDataset.concatenate_dataframes(
            dataframes=[
                pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'e': 5}, {'a': 2, 'b': 3, 'c': 4, 'e': 6}]),
                pd.DataFrame([{'a': 3, 'b': 4, 'c': 5, 'e': 7}, {'a': 4, 'b': 5, 'c': 6, 'e': 8}]),
                pd.DataFrame([{'a': 5, 'b': 6, 'c': 7, 'e': 9}, {'a': 6, 'b': 7, 'c': 8, 'e': 10}]),
            ],
            split_names=['first', 'second', 'third']
        ), npartitions=5, sort=True)


class SklearnPipelineMultiLabelDaskDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = BaseDaskDataset
    pipeline_cls = SklearnPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c', 'd'], 'other_named_split_sections': {'other': ['e']}, 'squeeze_return': False}

    @property
    def pipeline_params(self):
        '''
        Sklearn cannot implement a pipeline without transformers
        '''
        return {'transformers': [('step', Transformer())]}

    def example_split_name(self):
        return None

    def expected_split_contents(self):
        return Split(
            X=dd.from_pandas(pd.DataFrame([
                {'a': 1, 'b': 2},
                {'a': 2, 'b': 3},
                {'a': 3, 'b': 4},
                {'a': 4, 'b': 5},
                {'a': 5, 'b': 6},
                {'a': 6, 'b': 7}]),
                npartitions=5, sort=True),
            y=dd.from_pandas(pd.DataFrame([
                {'c': 3, 'd': 4},
                {'c': 4, 'd': 5},
                {'c': 5, 'd': 6},
                {'c': 6, 'd': 7},
                {'c': 7, 'd': 8},
                {'c': 8, 'd': 9}]),
                npartitions=5, sort=True),
            other=dd.from_pandas(pd.DataFrame([
                {'e': 5},
                {'e': 6},
                {'e': 7},
                {'e': 8},
                {'e': 9},
                {'e': 10}]),
                npartitions=5, sort=True)
        )

    def build_dataset(self):
        # use pandas concat to handle index deduplication
        return dd.from_pandas(BasePandasDataset.concatenate_dataframes(
            dataframes=[
                pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}, {'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6}]),
                pd.DataFrame([{'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7}, {'a': 4, 'b': 5, 'c': 6, 'd': 7, 'e': 8}]),
                pd.DataFrame([{'a': 5, 'b': 6, 'c': 7, 'd': 8, 'e': 9}, {'a': 6, 'b': 7, 'c': 8, 'd': 9, 'e': 10}]),
            ],
            split_names=['first', 'second', 'third']
        ), npartitions=5, sort=True)


class ExplicitSplitSklearnPipelineMultiLabelDaskDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = BaseDaskDataset
    pipeline_cls = ExplicitSplitSklearnPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c', 'd'], 'other_named_split_sections': {'other': ['e']}, 'squeeze_return': False}

    @property
    def pipeline_params(self):
        '''
        Sklearn cannot implement a pipeline without transformers
        '''
        return {'transformers': [('step', Transformer())]}

    def example_split_name(self):
        return 'first'

    def expected_split_contents(self):
        return Split(
            X=dd.from_pandas(pd.DataFrame([
                {'a': 1, 'b': 2},
                {'a': 2, 'b': 3}]),
                npartitions=5, sort=True),
            y=dd.from_pandas(pd.DataFrame([
                {'c': 3, 'd': 4},
                {'c': 4, 'd': 5}]),
                npartitions=5, sort=True),
            other=dd.from_pandas(pd.DataFrame([
                {'e': 5},
                {'e': 6}]),
                npartitions=5, sort=True)
        )

    def build_dataset(self):
        # use pandas concat to handle index deduplication
        return dd.from_pandas(BasePandasDataset.concatenate_dataframes(
            dataframes=[
                pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}, {'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6}]),
                pd.DataFrame([{'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7}, {'a': 4, 'b': 5, 'c': 6, 'd': 7, 'e': 8}]),
                pd.DataFrame([{'a': 5, 'b': 6, 'c': 7, 'd': 8, 'e': 9}, {'a': 6, 'b': 7, 'c': 8, 'd': 9, 'e': 10}]),
            ],
            split_names=['first', 'second', 'third']
        ), npartitions=5, sort=True)


class RandomSplitSklearnPipelineMultiLabelDaskDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = BaseDaskDataset
    pipeline_cls = RandomSplitSklearnPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c', 'd'], 'other_named_split_sections': {'other': ['e']}, 'squeeze_return': False}

    @property
    def pipeline_params(self):
        '''
        Sklearn cannot implement a pipeline without transformers
        '''
        return {'train_size': 0.5, 'random_state': 10, 'transformers': [('step', Transformer())]}

    def example_split_name(self):
        return 'TRAIN'

    def expected_split_contents(self):
        return Split(
            X=dd.from_pandas(pd.DataFrame([
                {'a': 4, 'b': 5},
                {'a': 5, 'b': 6},
                {'a': 2, 'b': 3}],
                index=[3, 4, 1]),
                npartitions=5, sort=True),
            y=dd.from_pandas(pd.DataFrame([
                {'c': 6, 'd': 7},
                {'c': 7, 'd': 8},
                {'c': 4, 'd': 5}],
                index=[3, 4, 1]),
                npartitions=5, sort=True),
            other=dd.from_pandas(pd.DataFrame([
                {'e': 8},
                {'e': 9},
                {'e': 6}],
                index=[3, 4, 1]),
                npartitions=5, sort=True)
        )

    def build_dataset(self):
        # use pandas concat to handle index deduplication
        return dd.from_pandas(BasePandasDataset.concatenate_dataframes(
            dataframes=[
                pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}, {'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6}]),
                pd.DataFrame([{'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7}, {'a': 4, 'b': 5, 'c': 6, 'd': 7, 'e': 8}]),
                pd.DataFrame([{'a': 5, 'b': 6, 'c': 7, 'd': 8, 'e': 9}, {'a': 6, 'b': 7, 'c': 8, 'd': 9, 'e': 10}]),
            ],
            split_names=['first', 'second', 'third']
        ), npartitions=5, sort=True)


'''
OrderedDict + Dask
'''


class OrderedDictPipelineSingleLabelDaskDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = BaseDaskDataset
    pipeline_cls = OrderedDictPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c'], 'other_named_split_sections': {'other': ['e']}, 'squeeze_return': True}

    def example_split_name(self):
        return None

    def expected_split_contents(self):
        return Split(
            X=dd.from_pandas(pd.DataFrame([
                {'a': 1, 'b': 2},
                {'a': 2, 'b': 3},
                {'a': 3, 'b': 4},
                {'a': 4, 'b': 5},
                {'a': 5, 'b': 6},
                {'a': 6, 'b': 7}]),
                npartitions=5, sort=True),
            y=dd.from_pandas(pd.Series([3, 4, 5, 6, 7, 8], name='c'),
                             npartitions=5, sort=True),
            other=dd.from_pandas(pd.Series([5, 6, 7, 8, 9, 10], name='e'),
                                 npartitions=5, sort=True))

    def build_dataset(self):
        # use pandas concat to handle index deduplication
        return dd.from_pandas(BasePandasDataset.concatenate_dataframes(
            dataframes=[
                pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'e': 5}, {'a': 2, 'b': 3, 'c': 4, 'e': 6}]),
                pd.DataFrame([{'a': 3, 'b': 4, 'c': 5, 'e': 7}, {'a': 4, 'b': 5, 'c': 6, 'e': 8}]),
                pd.DataFrame([{'a': 5, 'b': 6, 'c': 7, 'e': 9}, {'a': 6, 'b': 7, 'c': 8, 'e': 10}]),
            ],
            split_names=['first', 'second', 'third']
        ), npartitions=5, sort=True)


class ExplicitSplitOrderedDictPipelineSingleLabelDaskDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = BaseDaskDataset
    pipeline_cls = ExplicitSplitOrderedDictPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c'], 'other_named_split_sections': {'other': ['e']}, 'squeeze_return': True}

    def example_split_name(self):
        return 'first'

    def expected_split_contents(self):
        return Split(
            X=dd.from_pandas(pd.DataFrame([
                {'a': 1, 'b': 2},
                {'a': 2, 'b': 3}]),
                npartitions=5, sort=True),
            y=dd.from_pandas(pd.Series([3, 4], name='c'),
                             npartitions=5, sort=True),
            other=dd.from_pandas(pd.Series([5, 6], name='e'),
                                 npartitions=5, sort=True))

    def build_dataset(self):
        # use pandas concat to handle index deduplication
        return dd.from_pandas(BasePandasDataset.concatenate_dataframes(
            dataframes=[
                pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'e': 5}, {'a': 2, 'b': 3, 'c': 4, 'e': 6}]),
                pd.DataFrame([{'a': 3, 'b': 4, 'c': 5, 'e': 7}, {'a': 4, 'b': 5, 'c': 6, 'e': 8}]),
                pd.DataFrame([{'a': 5, 'b': 6, 'c': 7, 'e': 9}, {'a': 6, 'b': 7, 'c': 8, 'e': 10}]),
            ],
            split_names=['first', 'second', 'third']
        ), npartitions=5, sort=True)


class RandomSplitOrderedDictPipelineSingleLabelDaskDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = BaseDaskDataset
    pipeline_cls = RandomSplitOrderedDictPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c'], 'other_named_split_sections': {'other': ['e']}, 'squeeze_return': True}

    @property
    def pipeline_params(self):
        return {'train_size': 0.5, 'random_state': 10}

    def example_split_name(self):
        return 'TRAIN'

    def expected_split_contents(self):
        return Split(
            X=dd.from_pandas(pd.DataFrame([
                {'a': 4, 'b': 5},
                {'a': 5, 'b': 6},
                {'a': 2, 'b': 3}],
                index=[3, 4, 1]),
                npartitions=5, sort=True),
            y=dd.from_pandas(pd.Series([6, 7, 4], index=[3, 4, 1], name='c'),
                             npartitions=5, sort=True),
            other=dd.from_pandas(pd.Series([8, 9, 6], index=[3, 4, 1], name='e'),
                                 npartitions=5, sort=True))

    def build_dataset(self):
        # use pandas concat to handle index deduplication
        return dd.from_pandas(BasePandasDataset.concatenate_dataframes(
            dataframes=[
                pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'e': 5}, {'a': 2, 'b': 3, 'c': 4, 'e': 6}]),
                pd.DataFrame([{'a': 3, 'b': 4, 'c': 5, 'e': 7}, {'a': 4, 'b': 5, 'c': 6, 'e': 8}]),
                pd.DataFrame([{'a': 5, 'b': 6, 'c': 7, 'e': 9}, {'a': 6, 'b': 7, 'c': 8, 'e': 10}]),
            ],
            split_names=['first', 'second', 'third']
        ), npartitions=5, sort=True)


class OrderedDictPipelineMultiLabelDaskDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = BaseDaskDataset
    pipeline_cls = OrderedDictPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c', 'd'], 'other_named_split_sections': {'other': ['e']}, 'squeeze_return': False}

    def example_split_name(self):
        return None

    def expected_split_contents(self):
        return Split(
            X=dd.from_pandas(pd.DataFrame([
                {'a': 1, 'b': 2},
                {'a': 2, 'b': 3},
                {'a': 3, 'b': 4},
                {'a': 4, 'b': 5},
                {'a': 5, 'b': 6},
                {'a': 6, 'b': 7}]),
                npartitions=5, sort=True),
            y=dd.from_pandas(pd.DataFrame([
                {'c': 3, 'd': 4},
                {'c': 4, 'd': 5},
                {'c': 5, 'd': 6},
                {'c': 6, 'd': 7},
                {'c': 7, 'd': 8},
                {'c': 8, 'd': 9}]),
                npartitions=5, sort=True),
            other=dd.from_pandas(pd.DataFrame([
                {'e': 5},
                {'e': 6},
                {'e': 7},
                {'e': 8},
                {'e': 9},
                {'e': 10}]),
                npartitions=5, sort=True)
        )

    def build_dataset(self):
        # use pandas concat to handle index deduplication
        return dd.from_pandas(BasePandasDataset.concatenate_dataframes(
            dataframes=[
                pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}, {'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6}]),
                pd.DataFrame([{'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7}, {'a': 4, 'b': 5, 'c': 6, 'd': 7, 'e': 8}]),
                pd.DataFrame([{'a': 5, 'b': 6, 'c': 7, 'd': 8, 'e': 9}, {'a': 6, 'b': 7, 'c': 8, 'd': 9, 'e': 10}]),
            ],
            split_names=['first', 'second', 'third']
        ), npartitions=5, sort=True)


class ExplicitSplitOrderedDictPipelineMultiLabelDaskDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = BaseDaskDataset
    pipeline_cls = ExplicitSplitOrderedDictPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c', 'd'], 'other_named_split_sections': {'other': ['e']}, 'squeeze_return': False}

    def example_split_name(self):
        return 'first'

    def expected_split_contents(self):
        return Split(
            X=dd.from_pandas(pd.DataFrame([
                {'a': 1, 'b': 2},
                {'a': 2, 'b': 3}]),
                npartitions=5, sort=True),
            y=dd.from_pandas(pd.DataFrame([
                {'c': 3, 'd': 4},
                {'c': 4, 'd': 5}]),
                npartitions=5, sort=True),
            other=dd.from_pandas(pd.DataFrame([
                {'e': 5},
                {'e': 6}]),
                npartitions=5, sort=True)
        )

    def build_dataset(self):
        # use pandas concat to handle index deduplication
        return dd.from_pandas(BasePandasDataset.concatenate_dataframes(
            dataframes=[
                pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}, {'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6}]),
                pd.DataFrame([{'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7}, {'a': 4, 'b': 5, 'c': 6, 'd': 7, 'e': 8}]),
                pd.DataFrame([{'a': 5, 'b': 6, 'c': 7, 'd': 8, 'e': 9}, {'a': 6, 'b': 7, 'c': 8, 'd': 9, 'e': 10}]),
            ],
            split_names=['first', 'second', 'third']
        ), npartitions=5, sort=True)


class RandomSplitOrderedDictPipelineMultiLabelDaskDatasetSplitTests(_PipelineSplitTests, unittest.TestCase):
    '''
    Pandas datasets are able to return copies of splits
    in case of downstream inplace mutations

    Validate consistent and resilient behavior
    '''
    dataset_cls = BaseDaskDataset
    pipeline_cls = RandomSplitOrderedDictPipeline

    @property
    def dataset_params(self):
        return {'label_columns': ['c', 'd'], 'other_named_split_sections': {'other': ['e']}, 'squeeze_return': False}

    @property
    def pipeline_params(self):
        return {'train_size': 0.5, 'random_state': 10}

    def example_split_name(self):
        return 'TRAIN'

    def expected_split_contents(self):
        return Split(
            X=dd.from_pandas(pd.DataFrame([
                {'a': 4, 'b': 5},
                {'a': 5, 'b': 6},
                {'a': 2, 'b': 3}],
                index=[3, 4, 1]),
                npartitions=5, sort=True),
            y=dd.from_pandas(pd.DataFrame([
                {'c': 6, 'd': 7},
                {'c': 7, 'd': 8},
                {'c': 4, 'd': 5}],
                index=[3, 4, 1]),
                npartitions=5, sort=True),
            other=dd.from_pandas(pd.DataFrame([
                {'e': 8},
                {'e': 9},
                {'e': 6}],
                index=[3, 4, 1]),
                npartitions=5, sort=True)
        )

    def build_dataset(self):
        # use pandas concat to handle index deduplication
        return dd.from_pandas(BasePandasDataset.concatenate_dataframes(
            dataframes=[
                pd.DataFrame([{'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}, {'a': 2, 'b': 3, 'c': 4, 'd': 5, 'e': 6}]),
                pd.DataFrame([{'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7}, {'a': 4, 'b': 5, 'c': 6, 'd': 7, 'e': 8}]),
                pd.DataFrame([{'a': 5, 'b': 6, 'c': 7, 'd': 8, 'e': 9}, {'a': 6, 'b': 7, 'c': 8, 'd': 9, 'e': 10}]),
            ],
            split_names=['first', 'second', 'third']
        ), npartitions=5, sort=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
