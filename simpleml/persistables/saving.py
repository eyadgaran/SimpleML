'''
Module to define the mixins that support different persistence
patterns for external objects

- Dataframe saving (as tables in dedicated schema)
- Pickled Object saving
    - In database as a binary blob
    - To local filestore
- HDF5 object saving
    - In database as a binary blob
    - To local filestore
- Remote filestore saving
    - S3
    - Google Cloud
    - Azure
'''

__author__ = 'Elisha Yadgaran'


from simpleml.persistables.binary_blob import BinaryBlob
from simpleml.utils.system_path import PICKLED_FILESTORE_DIRECTORY, HDF5_FILESTORE_DIRECTORY
from simpleml.persistables.meta_registry import KERAS_REGISTRY
from abc import ABCMeta, abstractmethod
import dill as pickle
from os.path import join

# Python 2/3 compatibility
try:
    import cStringIO
except ImportError:
    from io import StringIO as cStringIO
from future.utils import with_metaclass

# Import optional dependencies
from simpleml import load_model, hickle


class BaseExternalSaveMixin(with_metaclass(ABCMeta, object)):

    @abstractmethod
    def _save_external_files(self):
        '''
        Define pattern for saving external files
        '''

    @abstractmethod
    def _load_external_files(self):
        '''
        Define pattern for loading external files

        should set the self._external_file attribute
        '''


class DataframeTableSaveMixin(BaseExternalSaveMixin):
    '''
    Mixin class to save dataframes to a database table

    Expects the following available attributes:
        - self._external_file
        - self._schema
        - self.id
        - self._engine
        - self.dataframe

    Sets the following attributes:
        - self.filepaths
    '''
    def _save_external_files(self):
        '''
        Unless overwritten only use this mixin's paradigm
        '''
        self._save_dataframe_to_table()

    def _load_external_files(self):
        '''
        Unless overwritten only use this mixin's paradigm
        '''
        self._load_dataframe_from_table()

    def _save_dataframe_to_table(self):
        '''
        Shared method to save dataframe into a new table with name = GUID
        '''
        self.filepaths = {"database": [(self._schema, str(self.id))]}
        self.df_to_sql(self._engine, self.dataframe,
                       str(self.id), schema=self._schema)

    def _load_dataframe_from_table(self):
        '''
        Shared method to load dataframe from database
        '''
        schema, tablename = self.filepaths['database'][0]
        self._external_file = self.load_sql(
            'select * from "{}"."{}"'.format(schema, tablename),
            self._engine
        )

        # Indicate externals were loaded
        self.unloaded_externals = False

    @staticmethod
    def df_to_sql(engine, df, table, dtype=None, schema='public',
                  if_exists='replace', sep='|', encoding='utf8', index=False):
        '''
        Utility to bulk insert pandas dataframe via `copy from`

        :param df: dataframe to insert
        :param table: destination table
        :param dtype: column schema of destination table
        :param schema: destination schema
        :param if_exists: what to do if destination table exists; valid inputs are:
        [`replace`, `append`, `fail`]
        :param sep: separator key between cells
        :param encoding: character encoding to use
        :param index: whether to output index with data
        '''

        # Create Table
        df.head(0).to_sql(table, con=engine, if_exists=if_exists,
                          index=index, schema=schema, dtype=dtype)

        # Prepare data
        output = cStringIO.StringIO()
        df.to_csv(output, sep=sep, header=False, encoding=encoding, index=index)
        output.seek(0)

        # Insert data
        connection = engine.raw_connection()
        cursor = connection.cursor()
        cursor.copy_from(output, '"' + '"."'.join([schema, table]) + '"', sep=sep, null='',
                         columns=['"{}"'.format(i) for i in df.columns])
        connection.commit()
        connection.close()


class DatabasePickleSaveMixin(BaseExternalSaveMixin):
    '''
    Mixin class to save binary objects to a database table

    Expects the following available attributes:
        - self._external_file
        - self.id
        - self.object_type

    Sets the following attributes:
        - self.filepaths
        - self.unloaded_externals
    '''
    def _save_external_files(self):
        '''
        Unless overwritten only use this mixin's paradigm
        '''
        self._save_pickle_to_database()

    def _load_external_files(self):
        '''
        Unless overwritten only use this mixin's paradigm
        '''
        self._load_pickle_from_database()

    def _save_pickle_to_database(self):
        '''
        Shared method to save files into binary schema

        Hardcoded to only store pickled objects in database so overwrite to use
        other storage mechanism
        '''
        pickled_file = pickle.dumps(self._external_file, protocol=pickle.HIGHEST_PROTOCOL)
        pickled_record = BinaryBlob.create(
            object_type=self.object_type, object_id=self.id, binary_blob=pickled_file)
        self.filepaths = {"database_pickled": [str(pickled_record.id)]}

    def _load_pickle_from_database(self):
        '''
        Shared method to load files from database

        Hardcoded to only pull from pickled so overwrite to use
        other storage mechanism
        '''
        pickled_id = self.filepaths['database_pickled'][0]
        pickled_file = BinaryBlob.find(pickled_id).binary_blob
        self._external_file = pickle.loads(pickled_file)

        # Indicate externals were loaded
        self.unloaded_externals = False


class DiskPickleSaveMixin(BaseExternalSaveMixin):
    '''
    Mixin class to save objects to disk in pickled format

    Expects the following available attributes:
        - self._external_file
        - self.id

    Sets the following attributes:
        - self.filepaths
        - self.unloaded_externals
    '''
    def _save_external_files(self):
        '''
        Unless overwritten only use this mixin's paradigm
        '''
        self._save_pickle_to_disk()

    def _load_external_files(self):
        '''
        Unless overwritten only use this mixin's paradigm
        '''
        self._load_pickle_from_disk()

    def _save_pickle_to_disk(self):
        '''
        Shared method to save files to disk in pickled format
        '''
        with open(join(PICKLED_FILESTORE_DIRECTORY, str(self.id)) + '.pkl', 'wb') as pickled_file:
            pickle.dump(self._external_file, pickled_file, protocol=pickle.HIGHEST_PROTOCOL)
        self.filepaths = {"disk_pickled": [str(self.id) + '.pkl']}

    def _load_pickle_from_disk(self):
        '''
        Shared method to load files from disk in pickled format
        '''
        pickled_id = self.filepaths['disk_pickled'][0]
        with open(join(PICKLED_FILESTORE_DIRECTORY, pickled_id), 'rb') as pickled_file:
            self._external_file = pickle.load(pickled_file)

        # Indicate externals were loaded
        self.unloaded_externals = False


class DiskHDF5SaveMixin(BaseExternalSaveMixin):
    '''
    Mixin class to save objects to disk in HDF5 format with hickle

    Expects the following available attributes:
        - self._external_file
        - self.id

    Sets the following attributes:
        - self.filepaths
        - self.unloaded_externals
    '''
    def _save_external_files(self):
        '''
        Unless overwritten only use this mixin's paradigm
        '''
        self._save_hdf5_to_disk()

    def _load_external_files(self):
        '''
        Unless overwritten only use this mixin's paradigm
        '''
        self._load_hdf5_from_disk()

    def _save_hdf5_to_disk(self):
        '''
        Shared method to save files to disk in hickle's HDF5 format
        '''
        filepath = join(HDF5_FILESTORE_DIRECTORY, str(self.id) + '.h5')
        hickle.dump(self._external_file, filepath, compression='gzip', compression_opts=9)
        self.filepaths = {"disk_hdf5": [str(self.id) + '.h5']}

    def _load_hdf5_from_disk(self):
        '''
        Shared method to load files from disk in hickle's HDF5 format
        '''
        object_id = self.filepaths['disk_hdf5'][0]
        self._external_file = hickle.load(join(HDF5_FILESTORE_DIRECTORY, object_id))

        # Indicate externals were loaded
        self.unloaded_externals = False


class KerasDiskHDF5SaveMixin(BaseExternalSaveMixin):
    '''
    Mixin class to save objects to disk in Keras's HDF5 format
    Keras's internal persistence mechanism utilizes HDF5 and implements a custom pattern

    Expects the following available attributes:
        - self._external_file
        - self.id

    Sets the following attributes:
        - self.filepaths
        - self.unloaded_externals
    '''
    def _save_external_files(self):
        '''
        Unless overwritten only use this mixin's paradigm
        '''
        self._save_keras_hdf5_to_disk()

    def _load_external_files(self):
        '''
        Unless overwritten only use this mixin's paradigm
        '''
        self._load_keras_hdf5_from_disk()

    def _save_keras_hdf5_to_disk(self):
        '''
        Shared method to save files to disk in Keras's HDF5 format
        '''
        filepath = join(HDF5_FILESTORE_DIRECTORY, str(self.id) + '.h5')
        self._external_file.save(filepath)
        self.filepaths = {"disk_keras_hdf5": [str(self.id) + '.h5']}

    def _load_keras_hdf5_from_disk(self):
        '''
        Shared method to load files from disk in Keras's HDF5 format
        '''
        object_id = self.filepaths['disk_keras_hdf5'][0]
        self._external_file = load_model(
            str(join(HDF5_FILESTORE_DIRECTORY, object_id)),
            custom_objects=KERAS_REGISTRY.registry)

        # Indicate externals were loaded
        self.unloaded_externals = False


class AllSaveMixin(DataframeTableSaveMixin, DatabasePickleSaveMixin, DiskPickleSaveMixin,
                   DiskHDF5SaveMixin, KerasDiskHDF5SaveMixin):
    def _save_external_files(self):
        '''
        Wrapper method around save mixins for different persistence patterns
        '''
        save_method = self.state['save_method']

        if save_method == 'database':
            self._save_dataframe_to_table()
        elif save_method == 'database_pickled':
            self._save_pickle_to_database()
        elif save_method == 'disk_pickled':
            self._save_pickle_to_disk()
        elif save_method == 'disk_hdf5':
            self._save_hdf5_to_disk()
        elif save_method == 'disk_keras_hdf5':
            self._save_keras_hdf5_to_disk()
        else:
            raise ValueError('Unsupported Save Method: {}'.format(save_method))

    def _load_external_files(self):
        '''
        Wrapper method around save mixins for different persistence patterns
        '''
        save_method = self.state['save_method']

        if save_method == 'database':
            self._load_dataframe_from_table()
        elif save_method == 'database_pickled':
            self._load_pickle_from_database()
        elif save_method == 'disk_pickled':
            self._load_pickle_from_disk()
        elif save_method == 'disk_hdf5':
            self._load_hdf5_from_disk()
        elif save_method == 'disk_keras_hdf5':
            self._load_keras_hdf5_from_disk()
        else:
            raise ValueError('Unsupported Load Method: {}'.format(save_method))
