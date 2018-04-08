from simpleml.datasets.base_dataset import BaseDataset


__author__ = 'Elisha Yadgaran'


class BaseRawDataset(BaseDataset):
    '''
    Base class for all Raw Dataset objects.

    -------
    Schema
    -------
    No additional columns
    '''

    __tablename__ = 'raw_datasets'
