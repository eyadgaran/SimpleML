'''
Module for different split methods for cross validation

    1) No Split -- Just use all the data
    2) Explicit Split -- dataset class defines the split
    3) Percentage -- random split support for train, validation, test
    4) Chronological -- time based split support for train, validation, test
    5) KFold
'''

__author__ = 'Elisha Yadgaran'


from simpleml.constants import TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT

from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split
from future.utils import with_metaclass
from collections import defaultdict
import pandas as pd


class Split(dict):
    '''
    Container class for splits
    '''

    def __getattr__(self, attr):
        '''
        Default attribute processor
        (Used in combination with __getitem__ to enable ** syntax)
        '''
        return self.get(attr, None)

    @staticmethod
    def is_null_type(obj):
        '''
        Helper to check for nulls - useful to not pass "empty" attributes
        so defaults of None will get returned downstream instead
        ex: **split -> all non null named params
        '''
        # NoneType
        if obj is None:
            return True

        # Pandas objects
        if isinstance(obj, (pd.DataFrame, pd.Series)) and obj.empty:
            return True

        # Empty built-ins - uses __nonzero__
        if isinstance(obj, (list, tuple, dict)) and not obj:
            return True

        # Else
        return False

    def squeeze(self):
        '''
        Helper method to clear up any null-type keys
        '''
        poppable_keys = [k for k, v in self.items() if self.is_null_type(v)]
        [self.pop(k) for k in poppable_keys]

        # Return self for easy chaining
        return self


class SplitContainer(defaultdict):
    '''
    Explicit instantiation of a defaultdict returning split objects
    '''

    def __init__(self, default_factory=Split, **kwargs):
        super(SplitContainer, self).__init__(default_factory, **kwargs)


class SplitMixin(with_metaclass(ABCMeta, object)):
    @abstractmethod
    def split_dataset(self):
        '''
        Set the split criteria

        Must set self._dataset_splits
        '''

    def containerize_split(self, split_dict):
        return SplitContainer(**split_dict)

    def get_split_names(self):
        if not hasattr(self, '_dataset_splits') or self._dataset_splits is None:
            self.split_dataset()
        return list(self._dataset_splits.keys())


class NoSplitMixin(SplitMixin):
    def split_dataset(self):
        '''
        Non-split mixin class. Returns full dataset for any split name
        '''
        default_split = Split(X=self.dataset.X, y=self.dataset.y).squeeze()
        self._dataset_splits = self.containerize_split({
            'default_factory': lambda: default_split
        })


class ExplicitSplitMixin(SplitMixin):
    def split_dataset(self):
        '''
        Method to split the dataframe into different sets. Assumes dataset
        explicitly delineates between train, validation, and test
        '''
        self._dataset_splits = self.containerize_split({
            TRAIN_SPLIT: Split(X=self.dataset.get('X', TRAIN_SPLIT), y=self.dataset.get('y', TRAIN_SPLIT)).squeeze(),
            VALIDATION_SPLIT: Split(X=self.dataset.get('X', VALIDATION_SPLIT), y=self.dataset.get('y', VALIDATION_SPLIT)).squeeze(),
            TEST_SPLIT: Split(X=self.dataset.get('X', TEST_SPLIT), y=self.dataset.get('y', TEST_SPLIT)).squeeze()
        })


class RandomSplitMixin(SplitMixin):
    '''
    Class to randomly split dataset into different sets
    '''

    def __init__(self, train_size, test_size=None, validation_size=0.0,
                 random_state=123, shuffle=True, **kwargs):
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

    def split_dataset(self):
        '''
        Overwrite method to split by percentage
        '''
        train_size = self.config.get('train_size')
        validation_size = self.config.get('validation_size')
        test_size = self.config.get('test_size')
        random_state = self.config.get('random_state')
        shuffle = self.config.get('shuffle')

        # Sklearn's train test split can only accomodate one split per iteration
        if test_size == 0:  # No split necessary
            X_remaining, y_remaining = self.dataset.X, self.dataset.y
            X_test, y_test = [], []
        else:
            X_remaining, X_test, y_remaining, y_test = train_test_split(
                self.dataset.X, self.dataset.y, test_size=test_size, random_state=random_state, shuffle=shuffle)

        calibrated_validation_size = float(validation_size) / (validation_size + train_size)
        if calibrated_validation_size == 0:  # No split necessary
            X_train, y_train = X_remaining, y_remaining
            X_val, y_val = [], []
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_remaining, y_remaining, test_size=calibrated_validation_size, random_state=random_state, shuffle=shuffle)

        self._dataset_splits = self.containerize_split({
            TRAIN_SPLIT: Split(X=X_train, y=y_train).squeeze(),
            VALIDATION_SPLIT: Split(X=X_val, y=y_val).squeeze(),
            TEST_SPLIT: Split(X=X_test, y=y_test).squeeze()
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
