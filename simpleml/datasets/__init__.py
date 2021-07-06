'''
Import modules to register class names in global registry

Define convenience classes composed of different mixins
'''

__author__ = 'Elisha Yadgaran'

import pandas as pd
import logging

from .base_dataset import Dataset
from .pandas_mixin import BasePandasDatasetMixin, SingleLabelPandasDatasetMixin, MultiLabelPandasDatasetMixin
from .numpy_mixin import NumpyDatasetMixin

from simpleml.utils.errors import DatasetError


LOGGER = logging.getLogger(__name__)


class _PandasDatasetPipelineBuildMixin(object):
    def build_dataframe(self) -> None:
        '''
        Transform raw dataset via dataset pipeline for production ready dataset
        Overwrite this method to disable raw dataset requirement
        '''
        if self.pipeline is None:
            raise DatasetError('Must set pipeline before building dataframe')

        split_names = self.pipeline.get_split_names()
        splits = [self.pipeline.transform(X=None, split=split_name) for split_name in split_names]
        merged_splits = [self.merge_split(split) for split in splits]

        if splits[0].y and not self.config['label_columns']:  # Propagate old labels to new dataset
            self.config['label_columns'] = splits[0].y.columns.tolist()

        if len(merged_splits) > 1:  # Combine multiple splits
            # Join row wise - drop index in case duplicates exist
            self.dataframe = pd.concat(merged_splits, axis=0, ignore_index=True)
        else:
            self.dataframe = merged_splits[0]


# Mixin implementations for convenience
class PandasDataset(Dataset, BasePandasDatasetMixin, _PandasDatasetPipelineBuildMixin):
    '''
    Composed mixin class with pandas helper methods and a predefined build
    routine, assuming dataset pipeline existence.

    WARNING: this class will fail if build_dataframe is not overwritten or a
    pipeline provided!
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # warn that this class is deprecated and will be  removed
        LOGGER.warn('PandasDataset class is deprecated and will be removed in a future release! Use `SingleLabelPandasDataset` or `MultiLabelPandasDataset` instead')


class SingleLabelPandasDataset(Dataset, SingleLabelPandasDatasetMixin, _PandasDatasetPipelineBuildMixin):
    '''
    Composed mixin class with pandas helper methods and a predefined build
    routine, assuming dataset pipeline existence.

    Expects labels to only be a single column (1 label per sample)

    WARNING: this class will fail if build_dataframe is not overwritten or a
    pipeline provided!
    '''


class MultiLabelPandasDataset(Dataset, MultiLabelPandasDatasetMixin, _PandasDatasetPipelineBuildMixin):
    '''
    Composed mixin class with pandas helper methods and a predefined build
    routine, assuming dataset pipeline existence.

    Expects multiple labels across many columns (multi labels per sample)

    WARNING: this class will fail if build_dataframe is not overwritten or a
    pipeline provided!
    '''


class NumpyDataset(Dataset, NumpyDatasetMixin):
    '''
    Composed mixin class with numpy helper methods and a predefined build
    routine, assuming dataset pipeline existence.

    WARNING: this class will fail if build_dataframe is not overwritten or a
    pipeline provided!
    '''

    def build_dataframe(self) -> None:
        '''
        Transform raw dataset via dataset pipeline for production ready dataset
        Overwrite this method to disable raw dataset requirement
        '''
        if self.pipeline is None:
            raise DatasetError('Must set pipeline before building dataframe')

        split_names = self.pipeline.get_split_names()
        splits = [(split_name, self.pipeline.transform(X=None, split=split_name)) for split_name in split_names]

        if splits[0][1].y and not self.config['label_columns']:
            # If there is a Y, explicitly label it
            self.config['label_columns'] = ['y']

        y_label = self.config['label_columns'][0]

        # Assume propagating logic since there is no clear way to join
        self._external_file = {
            split_name: {
                'X': split.X,
                y_label: split.y
            } for split_name, split in splits
        }
