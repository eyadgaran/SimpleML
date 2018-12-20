'''
Numpy Module for external "dataframe"

Inherit and extend for particular patterns. It is a bit of a misnomer to use the
term "dataframe", since there are very few expected attributes and they are by no
means unique to pandas.
'''

__author__ = 'Elisha Yadgaran'


class NumpyDatasetMixin(object):
    '''
    Assumes _external_file is a dictionary of numpy ndarrays
    '''
    @property
    def X(self):
        '''
        Return the subset that isn't in the target labels
        '''
        return self.dataframe.get('X')

    @property
    def y(self):
        '''
        Return the target label columns
        '''
        if self.label_columns:
            return self.dataframe.get(self.label_columns[0])
        return None

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
        return ['X']
