'''
Module for different split methods for cross validation

    1) Percentage -- random split support for train, validation, test
    2) Chronological -- time based split support for train, validation, test
    3) KFold
'''

__author__ = 'Elisha Yadgaran'


from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split


TRAIN_SPLIT = 'TRAIN'
VALIDATION_SPLIT = 'VALIDATION'
TEST_SPLIT = 'TEST'


class SplitMixin(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def split_dataset(self):
        '''
        Set the split criteria
        '''


class NoSplitMixin(SplitMixin):
    def split_dataset(self):
        '''
        Method to split the dataframe into different sets. By default sets
        everything to `TRAIN`, but can be overwritten to add validation, test...

        TODO: Work in support for generators (k-fold)
        '''
        return {
            TRAIN_SPLIT: (self.dataset.X, self.dataset.y),
            VALIDATION_SPLIT: (self.dataset.X.head(0), self.dataset.y.head(0)),
            TEST_SPLIT: (self.dataset.X.head(0), self.dataset.y.head(0))
        }


class RandomSplitMixin(SplitMixin):
    '''
    Class to randomly split dataset into different sets
    '''
    def __init__(self, train_size, test_size=None, validation_size=0.0,
                 random_state=123, **kwargs):
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
            'random_state': random_state
        })

    def split_dataset(self):
        '''
        Overwrite method to split by percentage
        '''
        train_size = self.config.get('train_size')
        validation_size = self.config.get('validation_size')
        test_size = self.config.get('test_size')
        random_state = self.config.get('random_state')

        # Sklearn's train test split can only accomodate one split per iteration
        X_remaining, X_test, y_remaining, y_test = train_test_split(
             self.dataset.X, self.dataset.y, test_size=test_size, random_state=random_state)

        calibrated_validation_size = float(validation_size) / (validation_size + train_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_remaining, y_remaining, test_size=calibrated_validation_size, random_state=random_state)

        return {
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
