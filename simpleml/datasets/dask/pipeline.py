'''
Pipeline derived datasets
'''

__author__ = 'Elisha Yadgaran'


from simpleml.utils.errors import DatasetError

from .base import BaseDaskDataset


class DaskPipelineDataset(BaseDaskDataset):
    '''
    Dask dataset class that generates the dataframe as the output of the
    linked pipeline
    '''

    def build_dataframe(self) -> None:
        '''
        Transform raw dataset via dataset pipeline for production ready dataset
        '''
        if self.pipeline is None:
            raise DatasetError('Must set pipeline before building dataframe')

        split_names = self.pipeline.get_split_names()
        splits = [self.pipeline.transform(X=None, split=split_name) for split_name in split_names]
        merged_splits = [self.merge_split(split) for split in splits]

        if len(merged_splits) > 1:  # Combine multiple splits
            # Join row wise - drop index in case duplicates exist
            self.dataframe = self.concatenate_dataframes(merged_splits, split_names)
        else:
            self.dataframe = merged_splits[0]
