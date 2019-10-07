'''
Module to define the mixins that support different persistence
patterns for external objects
Nomenclature -> Save Location : Save Format

- Database Storage
    - database_table: Dataframe saving (as tables in dedicated schema)
    - database_pickled: In database as a binary blob
    - database_hdf5: In database as a binary blob
- Local Filesystem Storage
    - disk_pickled: Pickled file on local disk
    - disk_hdf5: HDF5 file on local disk
    - disk_keras_hdf5: Keras formatted HDF5 file on local disk
- Cloud Storage
    - cloud_pickled: Pickled file on cloud backend
    - cloud_hdf5: HDF5 file on cloud backend
    - cloud_keras_hdf5: Keras formatted HDF5 file on cloud backend
  Supported Backends:
    - Amazon S3
    - Google Cloud Platform
    - Microsoft Azure
    - Microsoft Onedrive
    - Aurora
    - Backblaze B2
    - DigitalOcean Spaces
    - OpenStack Swift
  Backend is determined by `cloud_section` in the configuration file
- Remote filestore saving
    - SCP to remote server
'''

__author__ = 'Elisha Yadgaran'


from simpleml.utils.binary_blob import BinaryBlob
from simpleml.utils.dataset_storage import DatasetStorage, DATASET_SCHEMA
from simpleml.utils.configuration import PICKLED_FILESTORE_DIRECTORY,\
    HDF5_FILESTORE_DIRECTORY, PICKLE_DIRECTORY, HDF5_DIRECTORY, CONFIG, CLOUD_SECTION
from simpleml.persistables.meta_registry import KERAS_REGISTRY
from abc import ABCMeta, abstractmethod
import cloudpickle as pickle
from os.path import join, isfile

# Python 2/3 compatibility
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
from future.utils import with_metaclass

# Import optional dependencies
from simpleml.imports import load_model, hickle, onedrivesdk

# Shared Connections
ONEDRIVE_SECTION = 'onedrive'
ONEDRIVE_CONNECTION = {}
CLOUD_DRIVER = None


class ExternalSaveMixin(with_metaclass(ABCMeta, object)):
    '''
    Base Class with save methods
    Subclasses should define the saving and loading patterns
    '''
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
        NULL_STRING = 'SIMPLEML_NULL'

        # Create Table
        df.head(0).to_sql(table, con=engine, if_exists=if_exists,
                          index=index, schema=schema, dtype=dtype)

        # Prepare data
        output = StringIO()
        df.to_csv(output, sep=sep, header=False, encoding=encoding, index=index, na_rep=NULL_STRING)
        output.seek(0)

        # Insert data
        connection = engine.raw_connection()
        cursor = connection.cursor()
        # Use copy expert for CSV formatting (handles character escapes, copy_from does not)
        cursor.copy_expert(
            """COPY "{schema}"."{table}" ({columns}) FROM STDIN WITH (FORMAT CSV, NULL '{null}', DELIMITER '{sep}')""".format(
                schema=schema,
                table=table,
                columns=', '.join(['"{}"'.format(i) for i in df.columns]),
                null=NULL_STRING,
                sep=sep
            ),
            output
        )
        connection.commit()
        connection.close()

    @staticmethod
    def pickle_object(obj, filepath=None):
        '''
        Pickles an object to a string or to the filesystem. Assumes that a NULL
        filepath expects a serialized string returned

        Prepends path to SimpleML Pickle directory before saving. ONLY pass in
        a relative filepath from that location
        '''
        if filepath is None:  # Return string instead of saving to file
            return pickle.dumps(obj)  # , protocol=pickle.HIGHEST_PROTOCOL)

        with open(join(PICKLED_FILESTORE_DIRECTORY, filepath), 'wb') as pickled_file:
            pickle.dump(obj, pickled_file)  # , protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_pickled_object(filepath, stream=False):
        '''
        Loads an object from a serialized string or filesystem. When stream is
        True, it tries to load the file directly from the string.

        Prepends path to SimpleML Pickle directory before loading. ONLY pass in
        a relative filepath from that location
        '''
        if stream:
            return pickle.loads(filepath)

        with open(join(PICKLED_FILESTORE_DIRECTORY, filepath), 'rb') as pickled_file:
            return pickle.load(pickled_file)

    @staticmethod
    def hickle_object(obj, filepath):
        '''
        Serializes an object to the filesystem in HDF5 format.

        Prepends path to SimpleML HDF5 directory before saving. ONLY pass in
        a relative filepath from that location
        '''
        hickle_file = join(HDF5_FILESTORE_DIRECTORY, filepath)
        hickle.dump(obj, hickle_file, compression='gzip', compression_opts=9)

    @staticmethod
    def load_hickled_object(filepath):
        '''
        Loads an object from the filesystem.

        Prepends path to SimpleML HDF5 directory before loading. ONLY pass in
        a relative filepath from that location
        '''
        hickle_file = join(HDF5_FILESTORE_DIRECTORY, filepath)
        return hickle.load(hickle_file)

    @staticmethod
    def save_keras_object(obj, filepath):
        '''
        Serializes an object to the filesystem in Keras HDF5 format.

        Prepends path to SimpleML HDF5 directory before saving. ONLY pass in
        a relative filepath from that location
        '''
        obj.save(join(HDF5_FILESTORE_DIRECTORY, filepath))

    @staticmethod
    def load_keras_object(filepath):
        '''
        Loads a Keras object from the filesystem.

        Prepends path to SimpleML HDF5 directory before loading. ONLY pass in
        a relative filepath from that location
        '''
        return load_model(
            str(join(HDF5_FILESTORE_DIRECTORY, filepath)),
            custom_objects=KERAS_REGISTRY.registry)


class DatabaseTableSaveMixin(ExternalSaveMixin):
    '''
    Mixin class to save dataframes to a database table

    Expects the following available attributes:
        - self._external_file
        - self.id
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
        engine = DatasetStorage.metadata.bind
        self.df_to_sql(engine, self.dataframe,
                       str(self.id), schema=DATASET_SCHEMA)

        self.filepaths = {"database_table": [(DATASET_SCHEMA, str(self.id))]}

    def _load_dataframe_from_table(self):
        '''
        Shared method to load dataframe from database
        '''
        schema, tablename = self.filepaths['database_table'][0]
        engine = DatasetStorage.metadata.bind
        self._external_file = self.load_sql(
            'select * from "{}"."{}"'.format(schema, tablename),
            engine
        )

        # Indicate externals were loaded
        self.unloaded_externals = False


class DatabasePickleSaveMixin(ExternalSaveMixin):
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
        pickled_file = self.pickle_object(self._external_file, as_stream=True)
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
        self._external_file = self.load_pickled_object(pickled_file, stream=True)

        # Indicate externals were loaded
        self.unloaded_externals = False


class DiskPickleSaveMixin(ExternalSaveMixin):
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
        filepath = str(self.id) + '.pkl'
        self.pickle_object(self._external_file, filepath)
        self.filepaths = {"disk_pickled": [filepath]}

    def _load_pickle_from_disk(self):
        '''
        Shared method to load files from disk in pickled format
        '''
        filepath = self.filepaths['disk_pickled'][0]
        self._external_file = self.load_pickled_object(filepath)

        # Indicate externals were loaded
        self.unloaded_externals = False


class DiskHDF5SaveMixin(ExternalSaveMixin):
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
        filepath = str(self.id) + '.h5'
        self.hickle_object(self._external_file, filepath)
        self.filepaths = {"disk_hdf5": [filepath]}

    def _load_hdf5_from_disk(self):
        '''
        Shared method to load files from disk in hickle's HDF5 format
        '''
        filepath = self.filepaths['disk_hdf5'][0]
        self._external_file = self.load_hickled_object(filepath)

        # Indicate externals were loaded
        self.unloaded_externals = False


class KerasDiskHDF5SaveMixin(ExternalSaveMixin):
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
        filename = str(self.id) + '.h5'
        self.save_keras_object(self._external_file, filename)
        self.filepaths = {"disk_keras_hdf5": [filename]}

    def _load_keras_hdf5_from_disk(self):
        '''
        Shared method to load files from disk in Keras's HDF5 format
        '''
        filepath = self.filepaths['disk_keras_hdf5'][0]
        self._external_file = self.load_keras_object(filepath)

        # Indicate externals were loaded
        self.unloaded_externals = False


class OnedriveBase(ExternalSaveMixin):
    '''
    Base class to save/load objects to Microsoft Onedrive
    '''
    @property
    def client(self):
        if ONEDRIVE_CONNECTION.get('client', None) is None:
            self.authenticate_onedrive()
        try:
            # See if existing client is configured with a valid token
            from onedrivesdk.error import OneDriveError
            ONEDRIVE_CONNECTION.get('client').item(id='root').get()
        except OneDriveError:
            self.authenticate_onedrive()
        return ONEDRIVE_CONNECTION.get('client')

    @client.setter
    def client(self, value):
        global ONEDRIVE_CONNECTION
        ONEDRIVE_CONNECTION['client'] = value

    @property
    def onedrive_root_id(self):
        if ONEDRIVE_CONNECTION.get('onedrive_root_id', None) is None:
            self.create_onedrive_schema()
        return ONEDRIVE_CONNECTION.get('onedrive_root_id')

    @onedrive_root_id.setter
    def onedrive_root_id(self, value):
        global ONEDRIVE_CONNECTION
        ONEDRIVE_CONNECTION['onedrive_root_id'] = value

    @property
    def onedrive_filestore_id(self):
        if ONEDRIVE_CONNECTION.get('onedrive_filestore_id', None) is None:
            self.create_onedrive_schema()
        return ONEDRIVE_CONNECTION.get('onedrive_filestore_id')

    @onedrive_filestore_id.setter
    def onedrive_filestore_id(self, value):
        global ONEDRIVE_CONNECTION
        ONEDRIVE_CONNECTION['onedrive_filestore_id'] = value

    @property
    def onedrive_pickle_id(self):
        if ONEDRIVE_CONNECTION.get('onedrive_pickle_id', None) is None:
            self.create_onedrive_schema()
        return ONEDRIVE_CONNECTION.get('onedrive_pickle_id')

    @onedrive_pickle_id.setter
    def onedrive_pickle_id(self, value):
        global ONEDRIVE_CONNECTION
        ONEDRIVE_CONNECTION['onedrive_pickle_id'] = value

    @property
    def onedrive_hdf5_id(self):
        if ONEDRIVE_CONNECTION.get('onedrive_hdf5_id', None) is None:
            self.create_onedrive_schema()
        return ONEDRIVE_CONNECTION.get('onedrive_hdf5_id')

    @onedrive_hdf5_id.setter
    def onedrive_hdf5_id(self, value):
        global ONEDRIVE_CONNECTION
        ONEDRIVE_CONNECTION['onedrive_hdf5_id'] = value

    def authenticate_onedrive(self):
        '''
        Authenticate with Onedrive Oauth2
        '''
        from onedrivesdk.helpers.GetAuthCodeServer import get_auth_code

        section = CONFIG[ONEDRIVE_SECTION]
        redirect_uri = section.get('redirect_uri')
        client_secret = section.get('client_secret')
        client_id = section.get('client_id')
        scopes = section.getlist('scopes')

        client = onedrivesdk.get_default_client(client_id=client_id, scopes=scopes)
        auth_url = client.auth_provider.get_auth_url(redirect_uri)
        # Block thread until we have the code
        code = get_auth_code(auth_url, redirect_uri)
        # Finally, authenticate!
        client.auth_provider.authenticate(code, redirect_uri, client_secret)

        self.client = client

    def create_onedrive_schema(self, root_folder='SIMPLEML'):
        '''
        Assumes already authenticated and assignment of self.client
        Checks if folders are already present, creates if not
        '''
        def create_onedrive_folder(parent, folder_name):
            try:  # Check if folder exists
                return parent.children[folder_name].get().id
            except onedrivesdk.error.OneDriveError:
                # Create folder
                folder = onedrivesdk.Folder()
                item = onedrivesdk.Item()
                item.name = folder_name
                item.folder = folder
                return parent.children.add(item).id

        # Create expected folders if they aren't present
        root_folders = root_folder.split('/')  # Catch subfolders
        root_id = 'root'
        for root_folder in root_folders:
            root_id = create_onedrive_folder(self.client.item(id=root_id), root_folder)
        filestore_folder_id = create_onedrive_folder(self.client.item(id=root_id), 'filestore')
        pickle_folder_id = create_onedrive_folder(self.client.item(id=filestore_folder_id), PICKLE_DIRECTORY)
        hdf5_folder_id = create_onedrive_folder(self.client.item(id=filestore_folder_id), HDF5_DIRECTORY)

        # Save IDs for quick reference later
        self.onedrive_root_id = root_id
        self.onedrive_filestore_id = filestore_folder_id
        self.onedrive_pickle_id = pickle_folder_id
        self.onedrive_hdf5_id = hdf5_folder_id

    def upload_to_onedrive(self, bucket, filename):
        '''
        Upload any file from disk to onedrive

        Steps:
            1) Authenticate
            2) Create Schema
            3) Upload
        '''
        if bucket == 'pickle':
            bucket_id = self.onedrive_pickle_id
            filepath = join(PICKLED_FILESTORE_DIRECTORY, filename)
        else:
            bucket_id = self.onedrive_hdf5_id
            filepath = join(HDF5_FILESTORE_DIRECTORY, filename)

        self.client.item(id=bucket_id).children[filename].upload(filepath)

    def download_from_onedrive(self, bucket, filename):
        '''
        Download any file from onedrive to disk

        Steps:
            1) Authenticate
            2) Get Folder IDs
            3) Download
        '''
        if bucket == 'pickle':
            bucket_id = self.onedrive_pickle_id
            filepath = join(PICKLED_FILESTORE_DIRECTORY, filename)
        else:
            bucket_id = self.onedrive_hdf5_id
            filepath = join(HDF5_FILESTORE_DIRECTORY, filename)

        # Check if file was already downloaded
        if isfile(filepath):
            return
        self.client.item(id=bucket_id).children[filename].download(filepath)


class OnedrivePickleSaveMixin(OnedriveBase):
    '''
    Mixin class to save objects to Microsoft Onedrive in pickled format

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
        self._save_pickle_to_onedrive()

    def _load_external_files(self):
        '''
        Unless overwritten only use this mixin's paradigm
        '''
        self._load_pickle_from_onedrive()

    def _save_pickle_to_onedrive(self):
        '''
        Shared method to save files to disk in pickled format
        Then upload pickled file from disk to onedrive
        '''
        filename = str(self.id) + '.pkl'
        bucket = 'pickle'
        self.pickle_object(self._external_file, filename)
        self.upload_to_onedrive(bucket, filename)
        self.filepaths = {"cloud_pickled": [filename]}

    def _load_pickle_from_onedrive(self):
        '''
        Download pickled file from onedrive to disk
        Then load files from disk in pickled format
        '''
        filename = self.filepaths['cloud_pickled'][0]
        bucket = 'pickle'
        self.download_from_onedrive(bucket, filename)
        self._external_file = self.load_pickled_object(filename)

        # Indicate externals were loaded
        self.unloaded_externals = False


class OnedriveHDF5SaveMixin(OnedriveBase):
    '''
    Mixin class to save objects to Microsoft Onedrive in HDF5 format

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
        self._save_hdf5_to_onedrive()

    def _load_external_files(self):
        '''
        Unless overwritten only use this mixin's paradigm
        '''
        self._load_hdf5_from_onedrive()

    def _save_hdf5_to_onedrive(self):
        '''
        Shared method to save files to disk in HDF5 format
        Then upload HDF5 file from disk to onedrive
        '''
        filename = str(self.id) + '.h5'
        bucket = 'hdf5'
        self.hickle_object(self._external_file, filename)
        self.upload_to_onedrive(bucket, filename)
        self.filepaths = {"cloud_hdf5": [filename]}

    def _load_hdf5_from_onedrive(self):
        '''
        Download HDF5 file from onedrive to disk
        Then load files from disk in HDF5 format
        '''
        filename = self.filepaths['cloud_hdf5'][0]
        bucket = 'hdf5'
        self.download_from_onedrive(bucket, filename)
        self._external_file = self.load_hickled_object(filename)

        # Indicate externals were loaded
        self.unloaded_externals = False


class OnedriveKerasHDF5SaveMixin(OnedriveBase):
    '''
    Mixin class to save objects to Microsoft Onedrive in Keras HDF5 format

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
        self._save_keras_hdf5_to_onedrive()

    def _load_external_files(self):
        '''
        Unless overwritten only use this mixin's paradigm
        '''
        self._load_keras_hdf5_from_onedrive()

    def _save_keras_hdf5_to_onedrive(self):
        '''
        Shared method to save files to disk in Keras HDF5 format
        Then upload HDF5 file from disk to onedrive
        '''
        filename = str(self.id) + '.h5'
        bucket = 'hdf5'
        self.save_keras_object(self._external_file, filename)
        self.upload_to_onedrive(bucket, filename)
        self.filepaths = {"cloud_keras_hdf5": [filename]}

    def _load_keras_hdf5_from_onedrive(self):
        '''
        Download HDF5 file from onedrive to disk
        Then load files from disk in HDF5 format
        '''
        filename = self.filepaths['cloud_keras_hdf5'][0]
        bucket = 'hdf5'
        self.download_from_onedrive(bucket, filename)
        self._external_file = self.load_keras_object(filename)

        # Indicate externals were loaded
        self.unloaded_externals = False


class CloudBase(ExternalSaveMixin):
    '''
    Base class to save/load objects via Apache Libcloud

    Generic api for all cloud providers so naming convention is extremely important
    to follow in the config. Please reference libcloud documentation for supported
    input parameters

    ```
    [cloud]
    section = `name of the config section to use, ex: s3`

    [s3]
    param = value --> normal key:value syntax. match these to however they are referenced later, examples:
    key = abc123
    secret = superSecure
    region = us-east-1
    something_specific_to_s3 = s3_parameter
    --- Below are internally referenced SimpleML params ---
    driver = S3 --> this must be the Apache Libcloud provider (https://github.com/apache/libcloud/blob/trunk/libcloud/storage/types.py)
    connection_params = key,secret,region,something_specific_to_s3 --> this determines the key: value params passed to the constructor (it can be different for each provider)
    path = simpleml/specific/root --> similar to disk based home directory, cloud home directory will start relative to here
    container = simpleml --> the cloud bucket or container name
    ```

    How this gets used:
    ```
    from libcloud.storage.types import Provider
    from libcloud.storage.providers import get_driver

    cloud_section = CONFIG.get(CLOUD_SECTION, 'section')
    connection_params = CONFIG.getlist(cloud_section, 'connection_params')
    root_path = CONFIG.get(cloud_section, 'path', fallback='')

    driver_cls = get_driver(getattr(Provider, CONFIG.get(cloud_section, 'driver')))
    driver = driver_cls(**{param: CONFIG.get(cloud_section, param) for param in connection_params})
    container = driver.get_container(container_name=CONFIG.get(cloud_section, 'container'))
    extra = {'content_type': 'application/octet-stream'}

    obj = driver.upload_object(LOCAL_FILE_PATH,
                               container=container,
                               object_name=root_path + simpleml_folder_path + filename,
                               extra=extra)

    obj = driver.download_object(CLOUD_OBJECT,
                                 destination_path=LOCAL_FILE_PATH,
                                 overwrite_existing=True,
                                 delete_on_failure=True)
    ```
    '''
    @property
    def driver(self):
        global CLOUD_DRIVER
        if CLOUD_DRIVER is None:
            from libcloud.storage.types import Provider
            from libcloud.storage.providers import get_driver

            cloud_section = CONFIG.get(CLOUD_SECTION, 'section')
            connection_params = CONFIG.getlist(cloud_section, 'connection_params')

            driver_cls = get_driver(getattr(Provider, CONFIG.get(cloud_section, 'driver')))
            driver = driver_cls(**{param: CONFIG.get(cloud_section, param) for param in connection_params})

            CLOUD_DRIVER = driver
        return CLOUD_DRIVER

    def upload_to_cloud(self, folder, filename):
        '''
        Upload any file from disk to cloud
        '''
        cloud_section = CONFIG.get(CLOUD_SECTION, 'section')
        root_path = CONFIG.get(cloud_section, 'path', fallback='')
        container = self.driver.get_container(container_name=CONFIG.get(cloud_section, 'container'))
        extra = {'content_type': 'application/octet-stream'}

        if folder == 'pickle':
            filepath = join(PICKLED_FILESTORE_DIRECTORY, filename)
            object_name = join(root_path, PICKLE_DIRECTORY, filename)
        else:
            filepath = join(HDF5_FILESTORE_DIRECTORY, filename)
            object_name = join(root_path, HDF5_DIRECTORY, filename)

        self.driver.upload_object(filepath,
                                  container=container,
                                  object_name=object_name,
                                  extra=extra)

    def download_from_cloud(self, folder, filename):
        '''
        Download any file from cloud to disk
        '''
        cloud_section = CONFIG.get(CLOUD_SECTION, 'section')
        root_path = CONFIG.get(cloud_section, 'path', fallback='')
        container = CONFIG.get(cloud_section, 'container')

        if folder == 'pickle':
            filepath = join(PICKLED_FILESTORE_DIRECTORY, filename)
            # Check if file was already downloaded before initiating cloud connection
            if isfile(filepath):
                return
            obj = self.driver.get_object(
                container_name=container,
                object_name=join(root_path, PICKLE_DIRECTORY, filename))
        else:
            filepath = join(HDF5_FILESTORE_DIRECTORY, filename)
            # Check if file was already downloaded before initiating cloud connection
            if isfile(filepath):
                return
            obj = self.driver.get_object(
                container_name=container,
                object_name=join(root_path, HDF5_DIRECTORY, filename))

        self.driver.download_object(obj,
                                    destination_path=filepath,
                                    overwrite_existing=True,
                                    delete_on_failure=True)


class CloudPickleSaveMixin(CloudBase):
    '''
    Mixin class to save objects to Cloud in pickled format

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
        self._save_pickle_to_cloud()

    def _load_external_files(self):
        '''
        Unless overwritten only use this mixin's paradigm
        '''
        self._load_pickle_from_cloud()

    def _save_pickle_to_cloud(self):
        '''
        Shared method to save files to disk in pickled format
        Then upload pickled file from disk to cloud
        '''
        filename = str(self.id) + '.pkl'
        folder = 'pickle'
        self.pickle_object(self._external_file, filename)
        self.upload_to_cloud(folder, filename)
        self.filepaths = {"cloud_pickled": [filename]}

    def _load_pickle_from_cloud(self):
        '''
        Download pickled file from cloud to disk
        Then load files from disk in pickled format
        '''
        filename = self.filepaths['cloud_pickled'][0]
        folder = 'pickle'
        self.download_from_cloud(folder, filename)
        self._external_file = self.load_pickled_object(filename)

        # Indicate externals were loaded
        self.unloaded_externals = False


class CloudHDF5SaveMixin(CloudBase):
    '''
    Mixin class to save objects to Cloud in HDF5 format

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
        self._save_hdf5_to_cloud()

    def _load_external_files(self):
        '''
        Unless overwritten only use this mixin's paradigm
        '''
        self._load_hdf5_from_cloud()

    def _save_hdf5_to_cloud(self):
        '''
        Shared method to save files to disk in HDF5 format
        Then upload HDF5 file from disk to cloud
        '''
        filename = str(self.id) + '.h5'
        folder = 'hdf5'
        self.hickle_object(self._external_file, filename)
        self.upload_to_cloud(folder, filename)
        self.filepaths = {"cloud_hdf5": [filename]}

    def _load_hdf5_from_cloud(self):
        '''
        Download HDF5 file from cloud to disk
        Then load files from disk in HDF5 format
        '''
        filename = self.filepaths['cloud_hdf5'][0]
        folder = 'hdf5'
        self.download_from_cloud(folder, filename)
        self._external_file = self.load_hickled_object(filename)

        # Indicate externals were loaded
        self.unloaded_externals = False


class CloudKerasHDF5SaveMixin(CloudBase):
    '''
    Mixin class to save objects to Cloud in Keras HDF5 format

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
        self._save_keras_hdf5_to_cloud()

    def _load_external_files(self):
        '''
        Unless overwritten only use this mixin's paradigm
        '''
        self._load_keras_hdf5_from_cloud()

    def _save_keras_hdf5_to_cloud(self):
        '''
        Shared method to save files to disk in Keras HDF5 format
        Then upload HDF5 file from disk to cloud
        '''
        filename = str(self.id) + '.h5'
        folder = 'hdf5'
        self.save_keras_object(self._external_file, filename)
        self.upload_to_cloud(folder, filename)
        self.filepaths = {"cloud_keras_hdf5": [filename]}

    def _load_keras_hdf5_from_cloud(self):
        '''
        Download HDF5 file from cloud to disk
        Then load files from disk in HDF5 format
        '''
        filename = self.filepaths['cloud_keras_hdf5'][0]
        folder = 'hdf5'
        self.download_from_cloud(folder, filename)
        self._external_file = self.load_keras_object(filename)

        # Indicate externals were loaded
        self.unloaded_externals = False


class AllSaveMixin(DatabaseTableSaveMixin, DatabasePickleSaveMixin,
                   DiskPickleSaveMixin, DiskHDF5SaveMixin, KerasDiskHDF5SaveMixin,
                   OnedrivePickleSaveMixin, OnedriveHDF5SaveMixin, OnedriveKerasHDF5SaveMixin,
                   CloudPickleSaveMixin, CloudHDF5SaveMixin, CloudKerasHDF5SaveMixin):
    def _save_external_files(self):
        '''
        Wrapper method around save mixins for different persistence patterns
        '''
        save_method = self.state['save_method']

        if save_method == 'database_table':
            self._save_dataframe_to_table()
        elif save_method == 'database_pickled':
            self._save_pickle_to_database()
        elif save_method == 'disk_pickled':
            self._save_pickle_to_disk()
        elif save_method == 'disk_hdf5':
            self._save_hdf5_to_disk()
        elif save_method == 'disk_keras_hdf5':
            self._save_keras_hdf5_to_disk()
        elif save_method == 'cloud_pickled':
            cloud_section = CONFIG.get(CLOUD_SECTION, 'section')
            if cloud_section == ONEDRIVE_SECTION:
                self._save_pickle_to_onedrive()
            else:
                self._save_pickle_to_cloud()
        elif save_method == 'cloud_hdf5':
            cloud_section = CONFIG.get(CLOUD_SECTION, 'section')
            if cloud_section == ONEDRIVE_SECTION:
                self._save_hdf5_to_onedrive()
            else:
                self._save_hdf5_to_cloud()
        elif save_method == 'cloud_keras_hdf5':
            cloud_section = CONFIG.get(CLOUD_SECTION, 'section')
            if cloud_section == ONEDRIVE_SECTION:
                self._save_keras_hdf5_to_onedrive()
            else:
                self._save_keras_hdf5_to_cloud()
        else:
            raise ValueError('Unsupported Save Method: {}'.format(save_method))

    def _load_external_files(self):
        '''
        Wrapper method around save mixins for different persistence patterns
        '''
        save_method = self.state['save_method']

        if save_method == 'database_table':
            self._load_dataframe_from_table()
        elif save_method == 'database_pickled':
            self._load_pickle_from_database()
        elif save_method == 'disk_pickled':
            self._load_pickle_from_disk()
        elif save_method == 'disk_hdf5':
            self._load_hdf5_from_disk()
        elif save_method == 'disk_keras_hdf5':
            self._load_keras_hdf5_from_disk()
        elif save_method == 'cloud_pickled':
            cloud_section = CONFIG.get(CLOUD_SECTION, 'section')
            if cloud_section == ONEDRIVE_SECTION:
                self._load_pickle_from_onedrive()
            else:
                self._load_pickle_from_cloud()
        elif save_method == 'cloud_hdf5':
            cloud_section = CONFIG.get(CLOUD_SECTION, 'section')
            if cloud_section == ONEDRIVE_SECTION:
                self._load_hdf5_from_onedrive()
            else:
                self._load_hdf5_from_cloud()
        elif save_method == 'cloud_keras_hdf5':
            cloud_section = CONFIG.get(CLOUD_SECTION, 'section')
            if cloud_section == ONEDRIVE_SECTION:
                self._load_keras_hdf5_from_onedrive()
            else:
                self._load_keras_hdf5_from_cloud()
        else:
            raise ValueError('Unsupported Load Method: {}'.format(save_method))
