from simpleml.persistables.base_persistable import BasePersistable
from simpleml.persistables.saving import AllSaveMixin


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

    def __init__(self, has_external_files=True, **kwargs):
        # By default assume unsupervised so no targets
        label_columns = kwargs.pop('label_columns', [])

        super(BaseDataset, self).__init__(
            has_external_files=has_external_files, **kwargs)

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
        dataframe = self.dataframe
        config = self.config

        return self.custom_hasher((dataframe, config))
