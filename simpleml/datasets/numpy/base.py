'''
Numpy Module for external "dataframe"

Inherit and extend for particular patterns. It is a bit of a misnomer to use the
term "dataframe", since there are very few expected attributes and they are by no
means unique to pandas.
'''

__author__ = 'Elisha Yadgaran'

import numpy as np

from typing import Any, List

from simpleml.datasets.base_dataset import Dataset
from simpleml.utils.errors import DatasetError
from simpleml.pipelines.validation_split_mixins import Split


class BaseNumpyDataset(Dataset):
    '''
    Assumes _external_file is a dictionary of numpy ndarrays
    '''
    # TODO: rewrite class to index native numpy array directly

    @property
    def X(self) -> np.ndarray:
        '''
        Return the subset that isn't in the target labels
        '''
        return self.get(column='X', split=None)

    @property
    def y(self) -> np.ndarray:
        '''
        Return the target label columns
        '''
        return self.get(column='y', split=None)

    def get(self, column: str, split: str) -> np.ndarray:
        '''
        Explicitly split validation splits
        Assumes self.dataframe has a get method to return a dictionary of {'X': X, 'y': y}
        Uses self.label_columns if y is named something else -- only looks at first entry in list

        returns None for any combination of column/split that isn't present
        '''
        if column not in ('X', 'y'):
            raise ValueError('Only support columns: X & y')

        if split is None:  # Assumes there is no top level split
            split_dict = self.dataframe
        else:
            split_dict = self.dataframe.get(split)

        if split_dict is None:
            split_dict = {}  # Make compatible with return syntax

        if column == 'y':
            return split_dict.get(self.label_columns[0], None)

        else:
            return split_dict.get('X', None)

    def get_split_names(self) -> List[str]:
        '''
        Helper to expose the splits contained in the dataset
        '''
        # assumes dict like container
        return list(self.dataframe.keys())

    def get_feature_names(self) -> List[str]:
        '''
        Should return a list of the features in the dataset
        '''
        return ['X']
