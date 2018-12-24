'''
Abstract Module for external dataframes

Inherit and extend for particular patterns. It is a bit of a misnomer to use the
term "dataframe", since there are very few expected attributes and they are by no
means unique to pandas.

Does not actually implement behavior. Added for transparency into expected methods
'''

__author__ = 'Elisha Yadgaran'


class AbstractDatasetMixin(object):
    @property
    def X(self):
        '''
        Return the subset that isn't in the target labels
        '''
        raise NotImplementedError

    @property
    def y(self):
        '''
        Return the target label columns
        '''
        raise NotImplementedError

    def get(self, column, split):
        '''
        Unimplemented method to explicitly split X and y
        Must be implemented by subclasses
        '''
        raise NotImplementedError

    def get_feature_names(self):
        '''
        Should return a list of the features in the dataset
        '''
        raise NotImplementedError
