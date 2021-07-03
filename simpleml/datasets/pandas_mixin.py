'''
Pandas Module for external dataframes

Inherit and extend for particular patterns. It is a bit of a misnomer to use the
term "dataframe", since there are very few expected attributes and they are by no
means unique to pandas.
'''

__author__ = 'Elisha Yadgaran'

import pandas as pd

from typing import Any, List

from simpleml.datasets.abstract_mixin import AbstractDatasetMixin
from simpleml.utils.errors import DatasetError


DATAFRAME_SPLIT_COLUMN: str = 'DATASET_SPLIT'


class PandasDatasetMixin(AbstractDatasetMixin):
    '''
    "Pandas"esque mixin class with control mechanism for `self.dataframe` of
    type `dataframe`. Only assumes pandas syntax, not types, so should be compatible
    with pandas drop-in replacements.

    In particular:
        A - type of pd.DataFrame:
            - query()
            - columns
            - drop()
            - __getitem__()
            - squeeze()

        B - any other type:
            - get()
            - __getitem__()
            - squeeze(
    '''
    @property
    def X(self) -> Any:
        '''
        Return the subset that isn't in the target labels (across all potential splits)
        '''
        return self.get(column='X', split=None)

    @property
    def y(self) -> Any:
        '''
        Return the target label columns
        '''
        return self.get(column='y', split=None)

    @property
    def _dataframe(self) -> pd.DataFrame:
        '''
        Overwrite base behavior to return a copy of the data in case consumers
        attempt to mutate the data structure
        '''
        if isinstance(self._external_file, pd.DataFrame):
            # return a copy so mutations can happen inplace with memory efficient objects
            return self._external_file.copy()
        else:
            return self._external_file

    def get(self, column: str, split: str) -> Union[pd.Series, pd.DataFrame]:
        '''
        Explicitly split validation splits
        Assumes self.dataframe has a get method to return the dataframe associated with the split
        Uses self.label_columns to separate x and y columns inside the returned dataframe

        returns empty dataframe for missing combinations of column & split
        '''
        if column not in ('X', 'y', None):
            raise ValueError('Only support columns: X, y, None')

        if isinstance(self.dataframe, pd.DataFrame):
            if split is None:  # Return the full dataset (all splits)
                df = self.dataframe
            else:
                # query automatically returns a copy
                if DATAFRAME_SPLIT_COLUMN not in self.dataframe.columns:
                    raise DatasetError('Cannot retrieve dataset split `{split}` from dataframe without `{DATAFRAME_SPLIT_COLUMN}` column')
                df = self.dataframe.query("{}=='{}'".format(DATAFRAME_SPLIT_COLUMN, split))
            if DATAFRAME_SPLIT_COLUMN in df.columns:
                df.drop(DATAFRAME_SPLIT_COLUMN, inplace=True, axis=1)
        else:
            # in case the dataframe is contained in a dict or other itemgetter enclosure
            # expects type pd.DataFrame returned
            df = self.dataframe.get(split, None)
            if df is None:  # Make compatible with subscription syntax
                df = pd.DataFrame()
            else:
                # copy for mutable downstream operations
                # self.dataframe only returns a copy if type dataframe
                df = df.copy()

        if column is None:
            return df

        elif column == 'y':  # Squeeze to reduce dimensionality of return
            # inplace drop non label columns
            drop_columns = [col for col in df.columns if col not in self.label_columns]
            df.drop(drop_columns, axis=1, inplace=True)
            return df.squeeze()

        else:  # X
            df.drop(self.label_columns, axis=1, inplace=True)
            return df

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
