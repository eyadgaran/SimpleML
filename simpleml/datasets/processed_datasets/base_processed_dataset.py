from simpleml.datasets.base_dataset import BaseDataset


__author__ = 'Elisha Yadgaran'


class BaseProcessedDataset(BaseDataset):
    '''
    Base class for all Processed Dataset objects.

    -------
    Schema
    -------
    No additional columns
    '''

    __tablename__ = 'datasets'
