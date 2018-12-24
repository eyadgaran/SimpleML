'''
Numpy Module for external "dataframe"

Inherit and extend for particular patterns. It is a bit of a misnomer to use the
term "dataframe", since there are very few expected attributes and they are by no
means unique to pandas.
'''

__author__ = 'Elisha Yadgaran'


from simpleml.datasets.abstract_mixin import AbstractDatasetMixin


class NumpyDatasetMixin(AbstractDatasetMixin):
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
        Explicitly split validation splits
        Assumes self.dataframe has a get method to return a dictionary of {'X': X, 'y': y}
        Uses self.label_columns if y is named something else -- only looks at first entry in list
        '''
        if column not in ('X', 'y'):
            raise ValueError('Only support columns: X & y')

        split_dict = self.dataframe.get(split)

        if column == 'y':
            return split_dict.get(self.label_columns[0])

        else:
            return split_dict.get('X')

    def get_feature_names(self):
        '''
        Should return a list of the features in the dataset
        '''
        return ['X']
