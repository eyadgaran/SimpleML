'''
Pandas Module for external dataframes

Inherit and extend for particular patterns. It is a bit of a misnomer to use the
term "dataframe", since there are very few expected attributes and they are by no
means unique to pandas.
'''

__author__ = 'Elisha Yadgaran'

import pandas as pd


class PandasDatasetMixin(object):
    @property
    def X(self):
        '''
        Return the subset that isn't in the target labels
        '''
        return self.dataframe[self.dataframe.columns.difference(self.label_columns)]

    @property
    def y(self):
        '''
        Return the target label columns
        '''
        return self.dataframe[self.label_columns]

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
        return self.X.columns.tolist()

    @staticmethod
    def load_csv(filename, **kwargs):
        '''Helper method to read in a csv file'''
        return pd.read_csv(filename, **kwargs)

    @staticmethod
    def load_sql(query, connection, **kwargs):
        '''Helper method to read in sql data'''
        return pd.read_sql_query(query, connection, **kwargs)
