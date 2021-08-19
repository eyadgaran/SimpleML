'''
Module for different pipeline split methods for cross validation

    1) No Split -- Just use all the data
    2) Explicit Split -- dataset class defines the split
    3) Percentage -- random split support for train, validation, test
    4) Chronological -- time based split support for train, validation, test
    5) KFold
'''

__author__ = 'Elisha Yadgaran'

import pandas as pd

from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split
from future.utils import with_metaclass
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

from simpleml.constants import TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT
from simpleml.datasets.dataset_splits import Split, SplitContainer
from .projected_splits import IdentityProjectedDatasetSplit, IndexBasedProjectedDatasetSplit


class SplitMixin(with_metaclass(ABCMeta, object)):
    @abstractmethod
    def split_dataset(self):
        '''
        Set the split criteria

        Must set self._dataset_splits
        '''

    def containerize_split(self, split_dict: Dict[str, Split]) -> SplitContainer:
        return SplitContainer(**split_dict)

    def get_split_names(self) -> List[str]:
        if not hasattr(self, '_dataset_splits') or self._dataset_splits is None:
            self.split_dataset()
        return list(self._dataset_splits.keys())


class NoSplitMixin(SplitMixin):
    def split_dataset(self) -> None:
        '''
        Non-split mixin class. Returns full dataset for any split name
        '''
        default_split = IdentityProjectedDatasetSplit(dataset=self.dataset, split=None)
        self._dataset_splits = self.containerize_split({
            'default_factory': lambda: default_split
        })


class ExplicitSplitMixin(SplitMixin):
    def split_dataset(self) -> None:
        '''
        Method to split the dataframe into different sets. Assumes dataset
        explicitly delineates between train, validation, and test
        '''
        self._dataset_splits = self.containerize_split({
            TRAIN_SPLIT: IdentityProjectedDatasetSplit(dataset=self.dataset, split=TRAIN_SPLIT),
            VALIDATION_SPLIT: IdentityProjectedDatasetSplit(dataset=self.dataset, split=VALIDATION_SPLIT),
            TEST_SPLIT: IdentityProjectedDatasetSplit(dataset=self.dataset, split=TEST_SPLIT)
        })


class RandomSplitMixin(SplitMixin):
    '''
    Class to randomly split dataset into different sets
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

        if train_size is None:
            train_size = 1.0 - validation_size

        if test_size is None:
            test_size = 1.0 - train_size - validation_size

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
            return data.index

        else:
            # no named index, use a linear range
            return range(len(data))

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

        if test_size == 0:  # No split necessary
            test_indices = []
            remaining_indices = index
        else:
            remaining_indices, test_indices = train_test_split(
                index, test_size=test_size, random_state=random_state, shuffle=shuffle)

        calibrated_validation_size = float(validation_size) / (validation_size + train_size)
        if calibrated_validation_size == 0:  # No split necessary
            train_indices = remaining_indices
            validation_indices = []
        else:
            train_indices, validation_indices = train_test_split(
                remaining_indices, test_size=calibrated_validation_size, random_state=random_state, shuffle=shuffle)

        self._dataset_splits = self.containerize_split({
            TRAIN_SPLIT: IdentityProjectedDatasetSplit(dataset=self.dataset, split=TRAIN_SPLIT, indices=train_indices),
            VALIDATION_SPLIT: IdentityProjectedDatasetSplit(dataset=self.dataset, split=VALIDATION_SPLIT, indices=validation_indices),
            TEST_SPLIT: IdentityProjectedDatasetSplit(dataset=self.dataset, split=TEST_SPLIT, indices=test_indices)
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
