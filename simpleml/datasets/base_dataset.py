from simpleml.persistables.base_persistable import BasePersistable
from abc import abstractmethod


__author__ = 'Elisha Yadgaran'


class BaseDataset(BasePersistable):
    '''
    Base class for all Dataset objects.

    Every dataset has one dataframe associated with it and can be subdivided
    by inheriting classes (y column for supervised, train/test/validation splits, etc)

    Dataset storage is the final resulting dataframe so technically a dataset
    is uniquely determined by Dataset class + Dataset Pipeline

    -------
    Schema
    -------
    No additional columns
    '''

    __abstract__ = True

    def __init__(self, *args, **kwargs):
        super(BaseDataset, self).__init__(*args, **kwargs)
        self._dataframe = None

    @property
    def dataframe(self):
        # Return dataframe if generated, otherwise generate first
        if self._dataframe is None:
            self.build_dataframe()
        return self._dataframe

    @abstractmethod
    def build_dataframe(self):
        '''
        Must set self._dataframe
        '''
