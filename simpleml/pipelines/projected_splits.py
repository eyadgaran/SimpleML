'''
Module for dataset projection into pipelines. Defines transfer objects
returned from pipelines
'''

__author__ = 'Elisha Yadgaran'


from abc import ABCMeta, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from simpleml.datasets.base_dataset import Dataset
from simpleml.datasets.dataset_splits import Split
from simpleml.imports import ddDataFrame, ddSeries


class ProjectedDatasetSplit(metaclass=ABCMeta):
    '''
    Transfer object to pass dataset splits through pipelines

    Contains a reference to the dataset and internal logic to
    project the split (references the dataset on each call to
    avoid mutability issues)

    Wraps the normal Split object but delegates behavior so can be used
    interchangeably
    '''

    def __init__(self,
                 dataset: Dataset,
                 split: Optional[str]):
        self.dataset = dataset
        self.split = split

    @property
    def dataset_split(self) -> Split:
        '''
        Passthrough method to retrieve the raw split
        '''
        return self.dataset.get_split(split=self.split)

    @abstractmethod
    def apply_projection(self, dataset_split: Split) -> Split:
        '''
        Main method to apply projection logic on the dataset split
        Returns a new Split with the data subset
        '''

    @property
    def projected_split(self) -> Split:
        '''
        Wrapper property to retrieve the dataset split and manipulate into a
        projected split. Returns a split object already parsed
        '''
        return self.apply_projection(self.dataset_split)

    def __getattr__(self, attr):
        '''
        Passthrough to treat a projected split like a normal split
        '''
        return getattr(self.projected_split, attr)

    def __getitem__(self, item):
        return getattr(self, item)


class IdentityProjectedDatasetSplit(ProjectedDatasetSplit):
    '''
    Straight passthrough variety of projection (ie projected split == dataset split)
    '''

    def apply_projection(self, dataset_split: Split) -> Split:
        '''
        Identity return
        '''
        return dataset_split.squeeze()


class IndexBasedProjectedDatasetSplit(ProjectedDatasetSplit):
    '''
    Index based subset. Compatible with dataset splits that support indexing
    '''

    def __init__(self, indices, **kwargs):
        super().__init__(**kwargs)
        self.indices = indices

    @classmethod
    def indexing_method(cls, df, *args, **kwargs):
        '''
        Infer indexing method to use based on type
        '''
        if isinstance(df, (pd.DataFrame, pd.Series)):
            return cls.pandas_indexing(df, *args, **kwargs)

        elif isinstance(df, np.ndarray):
            return cls.numpy_indexing(df, *args, **kwargs)

        elif isinstance(df, (ddDataFrame, ddSeries)):
            return cls.dask_indexing(df, *args, **kwargs)

        else:
            raise NotImplementedError('Add additional indexing methods to support other dtypes')

    @staticmethod
    def dask_indexing(df, indices):
        # dask indexing requires known divisions
        # https://docs.dask.org/en/stable/dataframe-design.html#dataframe-design
        return df.loc[indices]

    @staticmethod
    def pandas_indexing(df, indices):
        return df.loc[indices]

    @staticmethod
    def numpy_indexing(df, indices):
        return df[indices]

    def apply_projection(self, dataset_split: Split) -> Split:
        '''
        Index subset return
        '''
        return Split(
            **{k: self.indexing_method(v, self.indices) for k, v in dataset_split.items()}
        ).squeeze()
