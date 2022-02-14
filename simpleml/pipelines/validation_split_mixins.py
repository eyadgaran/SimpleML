'''
Module for different pipeline split methods for cross validation

    1) No Split -- Just use all the data - hardcoded as the default for all pipelines
    2) Explicit Split -- dataset class defines the split
    3) Percentage -- random split support for train, validation, test
    4) Chronological -- time based split support for train, validation, test
    5) KFold
'''

__author__ = 'Elisha Yadgaran'

from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Union

from future.utils import with_metaclass

import pandas as pd
from simpleml.constants import TEST_SPLIT, TRAIN_SPLIT, VALIDATION_SPLIT
from simpleml.datasets.dataset_splits import Split, SplitContainer
from simpleml.imports import ddDataFrame, ddSeries
from sklearn.model_selection import train_test_split

from .projected_splits import (IdentityProjectedDatasetSplit,
                               IndexBasedProjectedDatasetSplit)


class SplitMixin(with_metaclass(ABCMeta, object)):
    @abstractmethod
    def split_dataset(self):
        '''
        Set the split criteria

        Must set self._dataset_splits
        '''

    def containerize_split(self, split_dict: Dict[str, Split]) -> SplitContainer:
        return SplitContainer(**split_dict)


class ExplicitSplitMixin(SplitMixin):
    def split_dataset(self) -> None:
        '''
        Method to split the dataframe into different sets. Assumes dataset
        explicitly delineates between different splits

        Passes forward dataset split names so uniquely named splits will propagate
        and can be referenced the same way
        '''
        self._dataset_splits = self.containerize_split({
            split_name: IdentityProjectedDatasetSplit(dataset=self.dataset, split=split_name)
            for split_name in self.dataset.get_split_names()
        })


class RandomSplitMixin(SplitMixin):
    '''
    Class to randomly split dataset into different sets

    **Redefines splits so custom named splits in dataset cannot be referenced
    by the same names. Only TRAIN/TEST/VALIDATION**
    '''

    def __init__(self,
                 train_size: Union[float, int],
                 test_size: Optional[Union[float, int]] = None,
                 validation_size: Union[float, int] = 0.0,
                 random_state: int = 123,
                 shuffle: bool = True, **kwargs):
        '''
        Set splitting params:
        By default validation is 0.0 because it is only used for hyperparameter
        tuning
        '''
        super(RandomSplitMixin, self).__init__(**kwargs)

        # Pipeline Params
        self.config.update({
            'train_size': train_size,
            'validation_size': validation_size,
            'test_size': test_size,
            'random_state': random_state,
            'shuffle': shuffle
        })

    @staticmethod
    def get_index(data) -> List[int]:
        '''
        Helper to extract the index from a dataset. Generates a range index
        if none exists
        '''
        if isinstance(data, (pd.DataFrame, pd.Series)):
            return data.index.tolist()
        elif isinstance(data, (ddDataFrame, ddSeries)):
            return data.index.compute().tolist()

        else:
            # no named index, use a linear range
            return list(range(len(data)))

    def split_dataset(self) -> None:
        '''
        Overwrite method to split by percentage
        '''
        train_size = self.config.get('train_size')
        validation_size = self.config.get('validation_size')
        test_size = self.config.get('test_size')
        random_state = self.config.get('random_state')
        shuffle = self.config.get('shuffle')

        # Sklearn's train test split can only accomodate one split per iteration
        # find the indices that match to each split
        # use the X split section
        index = self.get_index(self.dataset.X)

        # Convert all sizes to counts to make typing consistent
        def rate_to_count(i):
            if isinstance(i, float) and i <= 1.0 and i >= 0.0:
                return round(i * len(index))
            return i

        train_size: int = rate_to_count(train_size)
        validation_size: int = rate_to_count(validation_size)
        test_size: int = rate_to_count(test_size)

        if train_size is None:
            # everything not already allocated is attributed to train
            train_size = len(index) - (validation_size or 0) - (test_size or 0)

        if test_size is None:
            # remaining unallocated is attributed to test
            test_size = len(index) - (validation_size or 0) - train_size

        if validation_size is None:
            validation_size = len(index) - train_size - test_size

        if test_size == 0:  # No split necessary
            test_indices = []
            remaining_indices = index
        elif test_size == len(index):
            test_indices = index
            remaining_indices = []
        else:
            remaining_indices, test_indices = train_test_split(
                index, test_size=test_size, random_state=random_state, shuffle=shuffle)

        if validation_size == 0:  # No split necessary
            train_indices = remaining_indices
            validation_indices = []
        elif validation_size == len(remaining_indices):
            validation_indices = remaining_indices
            train_indices = []
        else:
            train_indices, validation_indices = train_test_split(
                remaining_indices, test_size=validation_size, random_state=random_state, shuffle=shuffle)

        self._dataset_splits = self.containerize_split({
            TRAIN_SPLIT: IndexBasedProjectedDatasetSplit(dataset=self.dataset, split=None, indices=train_indices),
            VALIDATION_SPLIT: IndexBasedProjectedDatasetSplit(dataset=self.dataset, split=None, indices=validation_indices),
            TEST_SPLIT: IndexBasedProjectedDatasetSplit(dataset=self.dataset, split=None, indices=test_indices)
        })


class ChronologicalSplitMixin(SplitMixin):
    def __init__(self, **kwargs):
        super(ChronologicalSplitMixin, self).__init__(**kwargs)


class KFoldSplitMixin(SplitMixin):
    '''
    TBD on how to implement this. KFold requires K models and unique datasets
    so may be easier to wrap a parallelized implementation that internally
    creates K new Pipeline and Model objects
    '''
    pass
