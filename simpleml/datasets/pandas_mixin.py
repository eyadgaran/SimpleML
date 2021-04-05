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

    def get(self, column: str, split: str) -> Any:
        '''
        Explicitly split validation splits
        Assumes self.dataframe has a get method to return the dataframe associated with the split
        Uses self.label_columns to separate x and y columns inside the returned dataframe

        returns empty dataframe for missing combinations of column & split
        '''
        if column not in ('X', 'y'):
            raise ValueError('Only support columns: X & y')

        if isinstance(self.dataframe, pd.DataFrame):
            if split is None:  # Return the full dataset (all splits)
                df = self.dataframe
            else:
                df = self.dataframe.query("{}=='{}'".format(DATAFRAME_SPLIT_COLUMN, split))
            if DATAFRAME_SPLIT_COLUMN in df.columns:
                df.drop(DATAFRAME_SPLIT_COLUMN, inplace=True, axis=1)
        else:
            df = self.dataframe.get(split)

        if df is None:  # Make compatible with subscription syntax
            df = pd.DataFrame()

        if column == 'y':  # Squeeze to reduce dimensionality of return
            return df[[col for col in self.label_columns if col in df.columns]].squeeze()

        else:
            return df[df.columns.difference(self.label_columns)]

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
