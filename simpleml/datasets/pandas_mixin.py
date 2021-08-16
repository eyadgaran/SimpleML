'''
Pandas Module for external dataframes

Inherit and extend for particular patterns. It is a bit of a misnomer to use the
term "dataframe", since there are very few expected attributes and they are by no
means unique to pandas.
'''

__author__ = 'Elisha Yadgaran'

import pandas as pd

from itertools import chain
from typing import List, Union, Optional

from simpleml.datasets.abstract_mixin import AbstractDatasetMixin
from simpleml.utils.errors import DatasetError
from simpleml.pipelines.validation_split_mixins import Split


DATAFRAME_SPLIT_COLUMN: str = 'DATASET_SPLIT'


class BasePandasDatasetMixin(AbstractDatasetMixin):
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

    WARNING: Needs to be used as a base class for datasets because it overwrites
    the standard dataset dataframe property
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

    def _validate_dtype(self, df: pd.DataFrame) -> None:
        '''
        Validating setter method for self._external_file
        Checks input is of type pd.DataFrame
        '''
        if not isinstance(df, pd.DataFrame):
            raise DatasetError('Pandas Datasets must be of type `pd.DataFrame`')

    def get(self, column: Optional[str], split: Optional[str]) -> pd.DataFrame:
        '''
        Explicitly split validation splits
        Uses self.label_columns to separate x and y columns inside the returned dataframe

        returns empty dataframe for missing combinations of column & split
        '''
        registered_sections = self.config.get('split_section_map')
        if column is not None and column != 'X' and column not in registered_sections:
            raise ValueError(f'Only support registered sections: {registered_sections}, X, or None')

        dataframe = self.dataframe  # copy

        # choose the columns to slice from the dataframe
        if column is None:  # All except internal columns
            return_columns = [col for col in dataframe.columns if col != DATAFRAME_SPLIT_COLUMN]

        elif column != 'X':
            # other passthrough columns
            return_columns = registered_sections[column]

        else:  # X
            all_other_columns = list(chain(*registered_sections.values()))
            return_columns = [
                col for col in dataframe.columns
                if col != DATAFRAME_SPLIT_COLUMN
                and col not in all_other_columns
            ]

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

    def get_split(self, split: Optional[str]) -> Split:
        '''
        Wrapper accessor to return a split object (for internal use)
        '''
        registered_sections = self.config.get('split_section_map')
        return Split(
            # explicitly get X as the "other" columns
            X=self.get(column='X', split=split),
            # should include y and any others if they exist
            **{section: self.get(split=split, column=section) for section in registered_sections}
        ).squeeze()

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

    @staticmethod
    def merge_split(split: Split) -> pd.DataFrame:
        '''
        Helper method to merge all dataframes in a split object into a single df
        does a column-wise join
        ex: `df1 = [A, B, C](4 rows)` + `df2 = [D, E, F](4 rows)`
        returns: `[A, B, C, D, E, F](4 rows)`
        '''
        return pd.concat(list(split.values()), axis=1)

    def get_feature_names(self) -> List[str]:
        '''
        Should return a list of the features in the dataset
        '''
        return self.X.columns.tolist()

    @staticmethod
    def load_csv(filename: str, **kwargs) -> pd.DataFrame:
        '''Helper method to read in a csv file'''
        return pd.read_csv(filename, **kwargs)

    @staticmethod
    def squeeze_dataframe(df: pd.DataFrame) -> pd.Series:
        '''
        Helper method to run dataframe squeeze and return a series
        '''
        return df.squeeze(axis=1)


class MultiLabelPandasDatasetMixin(BasePandasDatasetMixin):
    '''
    Multilabel implementation of pandas dataset - same as base for now
    '''
    pass


class SingleLabelPandasDatasetMixin(BasePandasDatasetMixin):
    '''
    Customized label logic for single label (y dimension = 1) datasets
    '''

    def _validate_schema(self, df: pd.DataFrame):
        '''
        Extend validation to check df has only a single column for the y section
        '''
        # validate single label status
        labels = self.label_columns
        if len(labels) != 1:
            raise DatasetError(f'SingleLabelPandasDataset requires exactly one label column, {len(labels)} found')

    @property
    def label_column(self):
        labels = self.label_columns

        # validate single label status
        if len(labels) != 1:
            raise DatasetError(f'SingleLabelPandasDataset requires exactly one label column, {len(labels)} found')

        return labels[0]

    def get(self, column: str, split: str) -> Union[pd.Series, pd.DataFrame]:
        '''
        Extends PandasDatasetMixin.get with logic to squeeze labels to a
        series (1D frame)
        '''
        data = super().get(column=column, split=split)

        if column == 'X':
            return data

        # Custom logic for other split sections
        # 1D dataframe can squeeze to a series
        return self.squeeze_dataframe(data)
