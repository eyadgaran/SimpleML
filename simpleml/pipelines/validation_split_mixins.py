'''
Module for different split methods for cross validation

    1) No Split -- Just use all the data
    2) Explicit Split -- dataset class defines the split
    3) Percentage -- random split support for train, validation, test
    4) Chronological -- time based split support for train, validation, test
    5) KFold
'''

__author__ = 'Elisha Yadgaran'


from simpleml import TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT

from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split
from future.utils import with_metaclass


class SplitMixin(with_metaclass(ABCMeta, object)):
    @abstractmethod
    def split_dataset(self):
        '''
        Set the split criteria

        Must set self._dataset_splits
        '''


class NoSplitMixin(SplitMixin):
    def split_dataset(self):
        '''
        Method to split the dataframe into different sets. By default sets
        everything to `TRAIN`, but can be overwritten to add validation, test...

        TODO: Work in support for generators (k-fold)
        '''
        self._dataset_splits = {
            TRAIN_SPLIT: (self.dataset.X, self.dataset.y),
            VALIDATION_SPLIT: (None, None),
            TEST_SPLIT: (None, None)
        }


class ExplicitSplitMixin(SplitMixin):
    def split_dataset(self):
        '''
        Method to split the dataframe into different sets. Assumes dataset
        explicitly delineates between train, validation, and test
        '''
        self._dataset_splits = {
            TRAIN_SPLIT: (self.dataset.get('X', TRAIN_SPLIT), self.dataset.get('y', TRAIN_SPLIT)),
            VALIDATION_SPLIT: (self.dataset.get('X', VALIDATION_SPLIT), self.dataset.get('y', VALIDATION_SPLIT)),
            TEST_SPLIT: (self.dataset.get('X', TEST_SPLIT), self.dataset.get('y', TEST_SPLIT))
        }


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
        X_remaining, X_test, y_remaining, y_test = train_test_split(
            self.dataset.X, self.dataset.y, test_size=test_size, random_state=random_state, shuffle=shuffle)

        calibrated_validation_size = float(validation_size) / (validation_size + train_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_remaining, y_remaining, test_size=calibrated_validation_size, random_state=random_state, shuffle=shuffle)

        self._dataset_splits = {
            TRAIN_SPLIT: (X_train, y_train),
            VALIDATION_SPLIT: (X_val, y_val),
            TEST_SPLIT: (X_test, y_test)
        }


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
