from simpleml.persistables.base_persistable import BasePersistable
from simpleml.persistables.saving import AllSaveMixin
import pandas as pd


__author__ = 'Elisha Yadgaran'


class BaseDataset(BasePersistable, AllSaveMixin):
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

    def __init__(self, has_external_files=True, save_method='database', **kwargs):
        # By default assume unsupervised so no targets
        label_columns = kwargs.pop('label_columns', [])

        super(BaseDataset, self).__init__(
            has_external_files=has_external_files, save_method=save_method, **kwargs)

        self.config['label_columns'] = label_columns
        self.object_type = 'DATASET'

        # Instantiate dataframe variable - doesn't get populated until
        # build_dataframe() is called
        self._external_file = None

    @property
    def dataframe(self):
        # Return dataframe if generated, otherwise generate first
        if self.unloaded_externals:
            self._load_external_files()

        if self._external_file is None:
            self.build_dataframe()

        return self._external_file

    @property
    def label_columns(self):
        '''
        Keep column list for labels in metadata to persist through saving
        '''
        return self.config.get('label_columns', [])

    @property
    def X(self):
        '''
        Return the subset that isn't in the target labels
        '''
        return self.dataframe[self.dataframe.columns.difference(self.label_columns)]

    @property
    def y(self):
        '''
        Return the target label columns
        '''
        return self.dataframe[self.label_columns]

    def build_dataframe(self):
        '''
        Must set self._external_file
        Cant set as abstractmethod because of database lookup dependency
        '''
        raise NotImplementedError

    def _hash(self):
        '''
        Datasets rely on external data so instead of hashing only the config,
        hash the actual resulting dataframe
        This requires loading the data before determining duplication
        so overwrite for differing behavior

        Hash is the combination of the:
            1) Dataframe
            2) Config
        '''
        return hash(self.custom_hasher((self.dataframe, self.config)))

    @staticmethod
    def load_csv(filename, **kwargs):
        '''Helper method to read in a csv file'''
        return pd.read_csv(filename, **kwargs)

    @staticmethod
    def load_sql(query, connection, **kwargs):
        '''Helper method to read in sql data'''
        return pd.read_sql_query(query, connection, **kwargs)
