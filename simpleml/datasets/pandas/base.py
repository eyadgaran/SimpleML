"""
Pandas Module for datasets

Inherit and extend for particular patterns
"""

__author__ = "Elisha Yadgaran"

from itertools import chain
from typing import List, Optional, Union

from simpleml.datasets.base_dataset import Dataset
from simpleml.pipelines.validation_split_mixins import Split
from simpleml.utils.errors import DatasetError

import pandas as pd

DATAFRAME_SPLIT_COLUMN: str = "DATASET_SPLIT"


class BasePandasDataset(Dataset):
    """
    Pandas base class with control mechanism for `self.dataframe` of
    type `pd.Dataframe`
    """

    def __init__(self, squeeze_return: bool = False, **kwargs):
        """
        :param squeeze_return: boolean flag whether to run dataframe.squeeze() on
            return from self.get() calls. Particularly necessary to align input
            types with different libraries (e.g. sklearn y with single label)
        """
        super().__init__(**kwargs)
        self.config["squeeze_return"] = squeeze_return

    @property
    def X(self) -> pd.DataFrame:
        """
        Return the subset that isn't in the target labels (across all potential splits)
        """
        return self.get(column="X", split=None)

    @property
    def y(self) -> pd.DataFrame:
        """
        Return the target label columns
        """
        return self.get(column="y", split=None)

    @property
    def _dataframe(self) -> pd.DataFrame:
        """
        Overwrite base behavior to return a copy of the data in case consumers
        attempt to mutate the data structure

        Only copies the pandas container - underlying cell objects can still propagate
        inplace mutations (eg lists, dicts, objects)
        """
        # return a copy so mutations can happen inplace with memory efficient objects
        return self._external_file.copy()

    def _validate_dtype(self, df: pd.DataFrame) -> None:
        """
        Validating setter method for self._external_file
        Checks input is of type pd.DataFrame
        """
        if not isinstance(df, pd.DataFrame):
            raise DatasetError("Pandas Datasets must be of type `pd.DataFrame`")

    def get(self, column: Optional[str], split: Optional[str]) -> pd.DataFrame:
        """
        Explicitly split validation splits
        Uses self.label_columns to separate x and y columns inside the returned dataframe

        returns empty dataframe for missing combinations of column & split
        """
        registered_sections = self.config.get("split_section_map")
        squeeze_return = self.config.get("squeeze_return")

        if column is not None and column != "X" and column not in registered_sections:
            raise ValueError(
                f"Only support registered sections: {registered_sections}, X, or None"
            )

        dataframe = self.dataframe  # copy

        # choose the columns to slice from the dataframe
        if column is None:  # All except internal columns
            return_columns = [
                col for col in dataframe.columns if col != DATAFRAME_SPLIT_COLUMN
            ]

        elif column != "X":
            # other passthrough columns
            return_columns = registered_sections[column]

        else:  # X
            all_other_columns = list(chain(*registered_sections.values()))
            return_columns = [
                col
                for col in dataframe.columns
                if col != DATAFRAME_SPLIT_COLUMN and col not in all_other_columns
            ]

        result = self._get(dataframe=dataframe, columns=return_columns, split=split)
        if squeeze_return:
            return self.squeeze_dataframe(result)
        else:
            return result

    @staticmethod
    def _get(dataframe: pd.DataFrame, columns: List[str], split: str) -> pd.DataFrame:
        """
        Internal method to extract data subsets from a dataframe

        :param dataframe: the dataframe to subset from
        :param columns: List of columns to slice from the dataframe
        :param split: row identifiers to slice rows (in internal column mapped to `DATAFRAME_SPLIT_COLUMN`)
        """
        if split is not None:  # Return the full dataset (all splits) - already a copy
            # query automatically returns a copy wisth a weakref
            if DATAFRAME_SPLIT_COLUMN not in dataframe.columns:
                raise DatasetError(
                    "Cannot retrieve dataset split `{split}` from dataframe without `{DATAFRAME_SPLIT_COLUMN}` column"
                )
            dataframe = dataframe.query(
                "{}=='{}'".format(DATAFRAME_SPLIT_COLUMN, split)
            )

        # inplace drop extra columns
        drop_columns = [col for col in dataframe.columns if col not in columns]
        if drop_columns:
            dataframe.drop(drop_columns, axis=1, inplace=True)

        # Last check in case any of the operations created a view or weakref copy
        if (hasattr(dataframe, "_is_view") and dataframe._is_view) or (
            hasattr(dataframe, "_is_copy") and dataframe._is_copy is not None
        ):
            dataframe = dataframe.copy()

        return dataframe

    def get_split(self, split: Optional[str]) -> Split:
        """
        Wrapper accessor to return a split object (for internal use)
        """
        registered_sections = self.config.get("split_section_map")
        return Split(
            # explicitly get X as the "other" columns
            X=self.get(column="X", split=split),
            # should include y and any others if they exist
            **{
                section: self.get(split=split, column=section)
                for section in registered_sections
            },
        ).squeeze()

    def get_split_names(self) -> List[str]:
        """
        Helper to expose the splits contained in the dataset
        """
        df = self.dataframe
        if DATAFRAME_SPLIT_COLUMN in df.columns:
            return df[DATAFRAME_SPLIT_COLUMN].unique().tolist()
        else:
            return []

    def get_feature_names(self) -> List[str]:
        """
        Should return a list of the features in the dataset
        """
        return self.X.columns.tolist()

    """
    Generic Pandas Helper Utils
    """

    @staticmethod
    def concatenate_dataframes(
        dataframes: List[pd.DataFrame], split_names: List[str]
    ) -> pd.DataFrame:
        """
        Helper method to merge dataframes into a single one with the split
        specified under `DATAFRAME_SPLIT_COLUMN`
        """
        for df, name in zip(dataframes, split_names):
            df[DATAFRAME_SPLIT_COLUMN] = name

        # Join row wise - drop index in case duplicates exist
        return pd.concat(dataframes, axis=0, ignore_index=True)

    @staticmethod
    def merge_split(split: Split) -> pd.DataFrame:
        """
        Helper method to merge all dataframes in a split object into a single df
        does a column-wise join
        ex: `df1 = [A, B, C](4 rows)` + `df2 = [D, E, F](4 rows)`
        returns: `[A, B, C, D, E, F](4 rows)`
        """
        return pd.concat(list(split.values()), axis=1)

    @staticmethod
    def squeeze_dataframe(df: pd.DataFrame) -> pd.Series:
        """
        Helper method to run dataframe squeeze and return a series
        """
        return df.squeeze(axis=1)
