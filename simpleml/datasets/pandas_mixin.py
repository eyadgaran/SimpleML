'''
Pandas Module for external dataframes

Inherit and extend for particular patterns. It is a bit of a misnomer to use the
term "dataframe", since there are very few expected attributes and they are by no
means unique to pandas.
'''

__author__ = 'Elisha Yadgaran'

import pandas as pd

from typing import List, Union

from simpleml.datasets.abstract_mixin import AbstractDatasetMixin
from simpleml.utils.errors import DatasetError


DATAFRAME_SPLIT_COLUMN: str = 'DATASET_SPLIT'


class PandasDatasetMixin(AbstractDatasetMixin):
    '''
    Pandas mixin class with control mechanism for `self.dataframe` of
    type `dataframe`. Mostly assumes pandas syntax, not types, so may be compatible
    with pandas drop-in replacements. Recommended to implement a parallel mixin
    for other frameworks though

    In particular:
        A - type of pd.DataFrame:
            - query()
            - columns
            - drop()
            - squeeze()
    '''
    @property
    def X(self) -> pd.DataFrame:
        '''
        Return the subset that isn't in the target labels (across all potential splits)
        '''
        return self.get(column='X', split=None)

    @property
    def y(self) -> pd.DataFrame:
        '''
        Return the target label columns
        '''
        return self.get(column='y', split=None)

    @property
    def _dataframe(self) -> pd.DataFrame:
        '''
        Overwrite base behavior to return a copy of the data in case consumers
        attempt to mutate the data structure

        Only copies the pandas container - underlying cell objects can still propagate
        inplace mutations (eg lists, dicts, objects)
        '''
        # return a copy so mutations can happen inplace with memory efficient objects
        return self._external_file.copy()

    @_dataframe.setter
    def _dataframe(self, df: pd.DataFrame) -> None:
        '''
        Validating setter method for self._external_file
        Checks input is of type pd.DataFrame
        '''
        if not isinstance(df, pd.DataFrame):
            raise DatasetError('Pandas Datasets must be of type `pd.DataFrame`')
        self._external_file = df

    def get(self, column: str, split: str) -> pd.DataFrame:
        '''
        Explicitly split validation splits
        Uses self.label_columns to separate x and y columns inside the returned dataframe

        returns empty dataframe for missing combinations of column & split
        '''
        if column not in ('X', 'y', None):
            raise ValueError('Only support columns: X, y, None')

        dataframe = self.dataframe  # copy

        # choose the columns to slice from the dataframe
        if column is None:  # All except internal columns
            return_columns = [col for col in dataframe.columns if col != DATAFRAME_SPLIT_COLUMN]

        elif column == 'y':  # Just label columns
            return_columns = self.label_columns

        else:  # X
            return_columns = [col for col in dataframe.columns if col != DATAFRAME_SPLIT_COLUMN and col not in self.label_columns]

        return self._get(dataframe=dataframe, columns=return_columns, split=split)

    @staticmethod
    def _get(dataframe: pd.DataFrame, columns: List[str], split: str) -> pd.DataFrame:
        '''
        Internal method to extract data subsets from a dataframe

        :param dataframe: the dataframe to subset from
        :param columns: List of columns to slice from the dataframe
        :param split: row identifiers to slice rows (in internal column mapped to `DATAFRAME_SPLIT_COLUMN`)
        '''
        if split is not None:  # Return the full dataset (all splits) - already a copy
            # query automatically returns a copy wisth a weakref
            if DATAFRAME_SPLIT_COLUMN not in dataframe.columns:
                raise DatasetError('Cannot retrieve dataset split `{split}` from dataframe without `{DATAFRAME_SPLIT_COLUMN}` column')
            dataframe = dataframe.query("{}=='{}'".format(DATAFRAME_SPLIT_COLUMN, split))

        # inplace drop extra columns
        drop_columns = [col for col in dataframe.columns if col not in columns]
        if drop_columns:
            dataframe.drop(drop_columns, axis=1, inplace=True)

        # Last check in case any of the operations created a view or weakref copy
        if (hasattr(dataframe, '_is_view') and dataframe._is_view) or \
                (hasattr(dataframe, '_is_copy') and dataframe._is_copy is not None):
            dataframe = dataframe.copy()

        return dataframe

    def concatenate_dataframes(self,
                               dataframes: List[pd.DataFrame],
                               split_names: List[str]) -> pd.DataFrame:
        '''
        Helper method to merge dataframes into a single one with the split
        specified under `DATAFRAME_SPLIT_COLUMN`
        '''
        for df, name in zip(dataframes, split_names):
            df[DATAFRAME_SPLIT_COLUMN] = name

        # Join row wise - drop index in case duplicates exist
        return pd.concat(dataframes, axis=0, ignore_index=True)

    def get_feature_names(self) -> List[str]:
        '''
        Should return a list of the features in the dataset
        '''
        return self.X.columns.tolist()

    @staticmethod
    def load_csv(filename: str, **kwargs) -> pd.DataFrame:
        '''Helper method to read in a csv file'''
        return pd.read_csv(filename, **kwargs)
