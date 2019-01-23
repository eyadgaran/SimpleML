'''
Import modules to register class names in global registry

Define convenience classes composed of different mixins
'''

__author__ = 'Elisha Yadgaran'


from .base_dataset import Dataset
from .pandas_mixin import PandasDatasetMixin
from .numpy_mixin import NumpyDatasetMixin

from simpleml.utils.errors import DatasetError

import pandas as pd


# Mixin implementations for convenience
class PandasDataset(Dataset, PandasDatasetMixin):
    def build_dataframe(self):
        '''
        Transform raw dataset via dataset pipeline for production ready dataset
        Overwrite this method to disable raw dataset requirement
        '''
        if self.pipeline is None:
            raise DatasetError('Must set pipeline before building dataframe')

        X, y = self.pipeline.transform(X=None, return_y=True)

        if y is None:
            y = pd.DataFrame()

        self.config['label_columns'] = y.columns.tolist()
        self._external_file = pd.concat([X, y], axis=1)


class NumpyDataset(Dataset, NumpyDatasetMixin):
    def build_dataframe(self):
        '''
        Transform raw dataset via dataset pipeline for production ready dataset
        Overwrite this method to disable raw dataset requirement
        '''
        if self.pipeline is None:
            raise DatasetError('Must set pipeline before building dataframe')

        X, y = self.pipeline.transform(X=None, return_y=True)

        if y is not None:
            self.config['label_columns'] = ['y']

        self._external_file = {'X': X, 'y': y}
