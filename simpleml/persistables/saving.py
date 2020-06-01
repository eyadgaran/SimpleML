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


import pandas as pd
from simpleml.utils.binary_blob import BinaryBlob
from simpleml.utils.dataset_storage import DatasetStorage, DATASET_SCHEMA
from simpleml.utils.configuration import PICKLED_FILESTORE_DIRECTORY,\
    HDF5_FILESTORE_DIRECTORY, PICKLE_DIRECTORY, HDF5_DIRECTORY, CONFIG, CLOUD_SECTION
from simpleml.utils.errors import SimpleMLError
from simpleml.persistables.meta_registry import KERAS_REGISTRY
import cloudpickle as pickle
from os.path import join, isfile
from typing import Optional, Any, Union, Callable, Dict, Type
from io import StringIO

# Import optional dependencies
from simpleml.imports import load_model, hickle, onedrivesdk

# Shared Connections
ONEDRIVE_SECTION = 'onedrive'
ONEDRIVE_CONNECTION = {}
CLOUD_DRIVER = None


class ExternalArtifactsMixin(object):
    '''
    Adds support for external artifacts. Contains decorators and internal registries
    for:
        - Artifacts
        - Save methods
        - Load methods

    This base class contains only staticmethods for interacting with different
    save location as well as the main methods for saving and loading using the
    registries.

    It is expected that this mixin will be used with a SimpleML persistable since
    it depends on the `self.filepaths` attribute

    The persistence paradigm:
        filepaths = {
            artifact_name: {
                save_pattern: filepaths: Any
            }
        }

    The nested notation is because any persistable can implement multiple save
    options (with arbitrary priority) and arbitrary inputs. Simple serialization
    could have only a single string location whereas complex artifacts might have
    a list or map of filepaths

    Keep a registry of the different save methods and the save/load functions
    that correspond. Avoiding a nested if/else block to allow subclasses to
    extend/overwrite without reimplementing
    SAVE_METHODS = {}
    LOAD_METHODS = {}
    Cannot implement the above lines in the base class because it propagates
    as a mutable object through different subclasses (ie: A -> B, A -> C,
    registering on C will also exist on A and B)
    Decorator is configured to create on first implementation and have all
    subclasses automatically inherit

    DEVELOPER BEWARE - Because of this behavior, combining multiple mixins (not
    subclassing) will overwrite the registry and require re-decoration
    ```
    class CombinedMixin(Mixin1, Mixin2,...):
    ```
    This will follow standard python inheritance and use the `Mixin1.SAVE_METHODS`
    as the resulting registry, discarding anything registered on any of the other
    classes
    '''

    class Decorators(object):
        '''
        Private decorators that can be used for registering methods for loading
        and saving.

        Contained in an internal class to enable decoration within the class
        (https://medium.com/@vadimpushtaev/decorator-inside-python-class-1e74d23107f6)
        '''
        @staticmethod
        def register_save_pattern(
            save_pattern: Optional[str] = None,
            save_method: Optional[str] = None,
            load_method: Optional[str] = None,
        ) -> Callable:
            '''
            Decorates a class to register the method(s) to use for saving and
            loading for the particular pattern

            :param save_pattern: the optional string denoting the pattern this
                class implements (e.g. `disk_pickled`)
            :param save_method: the optional string referencing the class method
                that is used for saving (`getattr(self, save_method)(...)`)
            :param load_method: the optional string referencing the class method
                that is used for loading (`getattr(self, load_method)(...)`)
            '''
            def register(cls: Type) -> Type:
                # Register the function name to be loaded with getattr(self, attribute)
                # Dont register the function directly to ensure the bound method gets
                # Called when invoked
                if not hasattr(cls, 'SAVE_METHODS'):
                    cls.SAVE_METHODS = {}

                if not hasattr(cls, 'LOAD_METHODS'):
                    cls.LOAD_METHODS = {}

                nonlocal save_pattern
                if save_pattern is None:
                    if not hasattr(cls, 'SAVE_PATTERN'):
                        raise SimpleMLError('Cannot register save pattern without passing the `save_pattern` parameter or setting the class attribute `cls.SAVE_PATTERN`')
                    save_pattern = cls.SAVE_PATTERN

                if save_method is not None:
                    cls.SAVE_METHODS[save_pattern] = save_method

                if load_method is not None:
                    cls.LOAD_METHODS[save_pattern] = load_method

                return cls
            return register

        @staticmethod
        def deregister_save_pattern(save_pattern: str) -> Callable:
            '''
            Class level decorator to deregister allowed saved patterns. Expects each class to
            implement as many as needed to accomodate.
            Expected to be used by subclasses that redefine patterns but dont
            want to expose the possibility of a developer accessing them.
            (By default registering patterns only exposes them to be persisted if
            declared in save_methods)
            '''
            def deregister(cls: Type) -> Type:
                if hasattr(cls, 'SAVE_METHODS'):
                    cls.SAVE_METHODS.pop(save_pattern)
                if hasattr(cls, 'LOAD_METHODS'):
                    cls.LOAD_METHODS.pop(save_pattern)
                return cls
            return deregister

        @staticmethod
        def register_artifact(artifact_name: str, save_attribute: str, restore_attribute: str) -> Callable:
            '''
            Class level decorator to define artifacts produced. Expects each class to
            implement as many as needed to accomodate.

            Format:
            ```
            @register_artifact(artifact_name='model', save_attribute='wrapper_attribute', restore_attribute='_internal_attribute')
            class NewPersistable(Persistable):
                @property
                def wrapper_attribute(self):
                    if not hasattr(self, _internal_attribute):
                        self._internal_attribute = self.create_attribute()
                    return self._internal_attribute
            ```
            Intentionally specify different attributes for saving and restoring
            to allow developer to wrap attribute in property decorator for
            lazy caching
            '''
            def register(cls: Type) -> Type:
                if not hasattr(cls, 'ARTIFACTS'):
                    cls.ARTIFACTS: Dict[str, Dict[str, str]] = {}
                cls.ARTIFACTS[artifact_name] = {'save': save_attribute, 'restore': restore_attribute}
                return cls
            return register

        @staticmethod
        def deregister_artifact(artifact_name: str) -> Callable:
            '''
            Class level decorator to deregister artifacts produced. Expects each class to
            implement as many as needed to accomodate.
            Expected to be used by subclasses that redefine artifacts but dont
            want to expose the possibility of a developer accessing them.
            (By default registering artifacts only exposes them to be persisted if
            declared in save_methods)
            '''
            def deregister(cls: Type) -> Type:
                if hasattr(cls, 'ARTIFACTS'):
                    cls.ARTIFACTS.pop(artifact_name)
                return cls
            return deregister

    def get_artifact(self, artifact_name: str) -> Any:
        '''
        Accessor method to lookup the artifact in the registry and return
        the corresponding data value
        '''
        if not hasattr(self, 'ARTIFACTS'):
            raise SimpleMLError('Cannot retrieve artifacts before registering. Make sure to decorate class with @ExternalArtifactsMixin.Decorators.register_artifact')
        if artifact_name not in self.ARTIFACTS:
            raise SimpleMLError(f'No registered artifact for {artifact_name}')
        save_attribute = self.ARTIFACTS[artifact_name]['save']
        return getattr(self, save_attribute)

    def restore_artifact(self, artifact_name: str, obj: Any) -> None:
        '''
        Setter method to lookup the restore attribute and set to the passed object
        '''
        if not hasattr(self, 'ARTIFACTS'):
            raise SimpleMLError('Cannot restore artifacts before registering. Make sure to decorate class with @ExternalArtifactsMixin.Decorators.register_artifact')
        if artifact_name not in self.ARTIFACTS:
            raise SimpleMLError(f'No registered artifact for {artifact_name}')
        restore_attribute = self.ARTIFACTS[artifact_name]['restore']
        setattr(self, restore_attribute, obj)

        # Make note that the artifact was loaded
        if hasattr(self, 'unloaded_artifacts'):
            try:
                self.unloaded_artifacts.remove(artifact_name)
            except ValueError:
                pass

    def save_external_file(self,
                           artifact_name: str, save_method: str,
                           **save_params) -> None:
        '''
        Abstracted pattern to save an artifact via one of the registered
        methods and update the filepaths location
        '''
        method = self.SAVE_METHODS.get(save_method, None)
        if method is None:
            raise SimpleMLError(f'No registered save pattern for {save_method}')
        filepath_data = getattr(self, method)(**save_params)

        # Update filepaths
        self.filepaths[artifact_name][save_method] = filepath_data

    def load_external_file(self, artifact_name: str, save_method: str) -> Any:
        '''
        Define pattern for loading external files
        returns the object for assignment
        Inverted operation from saving. Registered functions should take in
        the same data (in the same form) of what is saved in the filepath
        '''
        method = self.LOAD_METHODS.get(save_method, None)
        if method is None:
            raise SimpleMLError(f'No registered load pattern for {save_method}')

        # Do some validation in case attempting to load unsaved artifact
        artifact = self.filepaths.get('artifact_name', None)
        if artifact is None:
            raise SimpleMLError(f'No artifact saved for {artifact_name}')
        if save_method not in artifact:
            raise SimpleMLError(f'No artifact saved using save pattern {save_method} for {artifact_name}')

        filepath_data = artifact[save_method]
        return getattr(self, method)(filepath_data)

    @staticmethod
    def df_to_sql(engine, df: pd.DataFrame, table: str,
                  dtype: Optional[Dict[str, str]] = None,
                  schema: str = 'public', if_exists: str = 'replace',
                  sep: str = '|', encoding: str = 'utf8',
                  index: bool = False) -> None:
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
    def pickle_object(obj: Any, filepath: Optional[str]=None, overwrite: bool=True) -> Union[str, None]:
        '''
        Pickles an object to a string or to the filesystem. Assumes that a NULL
        filepath expects a serialized string returned

        Prepends path to SimpleML Pickle directory before saving. ONLY pass in
        a relative filepath from that location

        :param overwrite: Boolean indicating whether to first check if pickled
            object is already serialized. Defaults to not checking, but can be
            leverage by implementations that want the same artifact in multiple
            places
        '''
        if filepath is None:  # Return string instead of saving to file
            return pickle.dumps(obj)  # , protocol=pickle.HIGHEST_PROTOCOL)

        # Append the filepath to the pickle storage directory
        filepath = join(PICKLED_FILESTORE_DIRECTORY, filepath)

        if not overwrite:
            # Check if file was already serialized
            if isfile(filepath):
                return

        with open(filepath, 'wb') as pickled_file:
            pickle.dump(obj, pickled_file)  # , protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_pickled_object(filepath: str, stream: bool = False) -> Any:
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
    def hickle_object(obj: Any, filepath: str, overwrite: bool = True) -> None:
        '''
        Serializes an object to the filesystem in HDF5 format.

        Prepends path to SimpleML HDF5 directory before saving. ONLY pass in
        a relative filepath from that location

        :param overwrite: Boolean indicating whether to first check if HDF5
            object is already serialized. Defaults to not checking, but can be
            leverage by implementations that want the same artifact in multiple
            places
        '''
        # Append the filepath to the HDF5 storage directory
        hickle_file = join(HDF5_FILESTORE_DIRECTORY, filepath)
        if not overwrite:
            # Check if file was already serialized
            if isfile(hickle_file):
                return
        hickle.dump(obj, hickle_file, compression='gzip', compression_opts=9)

    @staticmethod
    def load_hickled_object(filepath: str) -> Any:
        '''
        Loads an object from the filesystem.

        Prepends path to SimpleML HDF5 directory before loading. ONLY pass in
        a relative filepath from that location
        '''
        hickle_file = join(HDF5_FILESTORE_DIRECTORY, filepath)
        return hickle.load(hickle_file)

    @staticmethod
    def save_keras_object(obj: Any, filepath: str, overwrite: bool = True) -> None:
        '''
        Serializes an object to the filesystem in Keras HDF5 format.

        Prepends path to SimpleML HDF5 directory before saving. ONLY pass in
        a relative filepath from that location

        :param overwrite: Boolean indicating whether to first check if HDF5
            object is already serialized. Defaults to not checking, but can be
            leverage by implementations that want the same artifact in multiple
            places
        '''
        # Append the filepath to the HDF5 storage directory
        hdf5_file = join(HDF5_FILESTORE_DIRECTORY, filepath)
        if not overwrite:
            # Check if file was already serialized
            if isfile(hdf5_file):
                return
        obj.save(hdf5_file)

    @staticmethod
    def load_keras_object(filepath: str) -> Any:
        '''
        Loads a Keras object from the filesystem.

        Prepends path to SimpleML HDF5 directory before loading. ONLY pass in
        a relative filepath from that location
        '''
        return load_model(
            str(join(HDF5_FILESTORE_DIRECTORY, filepath)),
            custom_objects=KERAS_REGISTRY.registry)


@ExternalArtifactsMixin.Decorators.register_save_pattern(
    save_method='_save_dataframe_to_table', load_method='_load_dataframe_from_table')
class DatabaseTableSaveMixin(ExternalArtifactsMixin):
    '''
    Mixin class to save dataframes to a database table
    '''
    SAVE_PATTERN = 'database_table'

    @classmethod
    def _save_dataframe_to_table(cls, obj: pd.DataFrame, persistable_id: str,
                                 schema: str = DATASET_SCHEMA,
                                 **kwargs) -> Dict[str, str]:
        '''
        Shared method to save dataframe into a new table with name = GUID
        Updates filepath for the artifact with the schema and table
        '''
        engine = DatasetStorage.metadata.bind
        cls.df_to_sql(engine, df=obj, table=persistable_id, schema=schema)

        return {'schema': schema, 'table': persistable_id}

    @classmethod
    def _load_dataframe_from_table(cls, filepath_data: Dict[str, str], **kwargs) -> pd.DataFrame:
        '''
        Shared method to load dataframe from database
        '''
        schema = filepath_data['schema']
        table = filepath_data['table']
        engine = DatasetStorage.metadata.bind
        df = cls.load_sql(
            'select * from "{}"."{}"'.format(schema, table),
            engine
        )

        return df


@ExternalArtifactsMixin.Decorators.register_save_pattern(
    save_method='_save_pickle_to_database', load_method='_load_pickle_from_database')
class DatabasePickleSaveMixin(ExternalArtifactsMixin):
    '''
    Mixin class to save binary objects to a database table
    '''
    SAVE_PATTERN = 'database_pickled'

    @classmethod
    def _save_pickle_to_database(cls, obj: Any, persistable_type: str,
                                 persistable_id: str, **kwargs) -> str:
        '''
        Shared method to save files into binary schema

        Hardcoded to only store pickled objects in database so overwrite to use
        other storage mechanism
        '''
        pickled_stream = cls.pickle_object(obj, as_stream=True)
        pickled_record = BinaryBlob.create(
            object_type=persistable_type, object_id=persistable_id, binary_blob=pickled_stream)
        return str(pickled_record.id)

    @classmethod
    def _load_pickle_from_database(cls, primary_key: str, **kwargs) -> Any:
        '''
        Shared method to load files from database

        Hardcoded to only pull from pickled so overwrite to use
        other storage mechanism
        '''
        pickled_stream = BinaryBlob.find(primary_key).binary_blob
        return cls.load_pickled_object(pickled_stream, stream=True)


@ExternalArtifactsMixin.Decorators.register_save_pattern(
    save_method='_save_pickle_to_disk', load_method='_load_pickle_from_disk')
class DiskPickleSaveMixin(ExternalArtifactsMixin):
    '''
    Mixin class to save objects to disk in pickled format
    '''
    SAVE_PATTERN = 'disk_pickled'

    @classmethod
    def _save_pickle_to_disk(cls, obj: Any, persistable_id: str, **kwargs) -> str:
        '''
        Shared method to save files to disk in pickled format
        '''
        filename = f'{persistable_id}.pkl'
        cls.pickle_object(obj, filename)
        return filename

    @classmethod
    def _load_pickle_from_disk(cls, filename: str, **kwargs) -> Any:
        '''
        Shared method to load files from disk in pickled format
        '''
        return cls.load_pickled_object(filename)


@ExternalArtifactsMixin.Decorators.register_save_pattern(
    save_method='_save_hdf5_to_disk', load_method='_load_hdf5_from_disk')
class DiskHDF5SaveMixin(ExternalArtifactsMixin):
    '''
    Mixin class to save objects to disk in HDF5 format with hickle
    '''
    SAVE_PATTERN = 'disk_hdf5'

    @classmethod
    def _save_hdf5_to_disk(cls, obj: Any, persistable_id: str, **kwargs) -> str:
        '''
        Shared method to save files to disk in hickle's HDF5 format
        '''
        filename = f'{persistable_id}.h5'
        cls.hickle_object(obj, filename)
        return filename

    @classmethod
    def _load_hdf5_from_disk(cls, filename: str, **kwargs) -> Any:
        '''
        Shared method to load files from disk in hickle's HDF5 format
        '''
        return cls.load_hickled_object(filename)


@ExternalArtifactsMixin.Decorators.register_save_pattern(
    save_method='_save_keras_hdf5_to_disk', load_method='_load_keras_hdf5_from_disk')
class KerasDiskHDF5SaveMixin(ExternalArtifactsMixin):
    '''
    Mixin class to save objects to disk in Keras's HDF5 format
    Keras's internal persistence mechanism utilizes HDF5 and implements a custom pattern
    '''
    SAVE_PATTERN = 'disk_keras_hdf5'

    @classmethod
    def _save_keras_hdf5_to_disk(cls, obj: Any, persistable_id: str, **kwargs) -> str:
        '''
        Shared method to save files to disk in Keras's HDF5 format
        '''
        filename = f'{persistable_id}.h5'
        cls.save_keras_object(obj, filename)
        return filename

    @classmethod
    def _load_keras_hdf5_from_disk(cls, filename: str, **kwargs) -> Any:
        '''
        Shared method to load files from disk in Keras's HDF5 format
        '''
        return cls.load_keras_object(filename)


class OnedriveBase(ExternalArtifactsMixin):
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


@ExternalArtifactsMixin.Decorators.register_save_pattern(
    save_method='_save_pickle_to_onedrive', load_method='_load_pickle_from_onedrive')
class OnedrivePickleSaveMixin(OnedriveBase):
    '''
    Mixin class to save objects to Microsoft Onedrive in pickled format
    '''
    SAVE_PATTERN = 'onedrive_pickled'

    @classmethod
    def _save_pickle_to_onedrive(cls, obj: Any, persistable_id: str, **kwargs) -> str:
        '''
        Shared method to save files to disk in pickled format
        Then upload pickled file from disk to onedrive
        '''
        filename = f'{persistable_id}.pkl'
        bucket = 'pickle'
        cls.pickle_object(obj, filename)
        cls.upload_to_onedrive(bucket, filename)
        return filename

    @classmethod
    def _load_pickle_from_onedrive(cls, filename: str, **kwargs) -> Any:
        '''
        Download pickled file from onedrive to disk
        Then load files from disk in pickled format
        '''
        bucket = 'pickle'
        cls.download_from_onedrive(bucket, filename)
        return cls.load_pickled_object(filename)


@ExternalArtifactsMixin.Decorators.register_save_pattern(
    save_method='_save_hdf5_to_onedrive', load_method='_load_hdf5_from_onedrive')
class OnedriveHDF5SaveMixin(OnedriveBase):
    '''
    Mixin class to save objects to Microsoft Onedrive in HDF5 format
    '''
    SAVE_PATTERN = 'onedrive_hdf5'

    @classmethod
    def _save_hdf5_to_onedrive(cls, obj: Any, persistable_id: str, **kwargs) -> str:
        '''
        Shared method to save files to disk in HDF5 format
        Then upload HDF5 file from disk to onedrive
        '''
        filename = f'{persistable_id}.h5'
        bucket = 'hdf5'
        cls.hickle_object(obj, filename)
        cls.upload_to_onedrive(bucket, filename)
        return filename

    @classmethod
    def _load_hdf5_from_onedrive(cls, filename: str, **kwargs) -> Any:
        '''
        Download HDF5 file from onedrive to disk
        Then load files from disk in HDF5 format
        '''
        bucket = 'hdf5'
        cls.download_from_onedrive(bucket, filename)
        return cls.load_hickled_object(filename)


@ExternalArtifactsMixin.Decorators.register_save_pattern(
    save_method='_save_keras_hdf5_to_onedrive', load_method='_load_keras_hdf5_from_onedrive')
class OnedriveKerasHDF5SaveMixin(OnedriveBase):
    '''
    Mixin class to save objects to Microsoft Onedrive in Keras HDF5 format
    '''
    SAVE_PATTERN = 'onedrive_keras_hdf5'

    @classmethod
    def _save_keras_hdf5_to_onedrive(cls, obj: Any, persistable_id: str, **kwargs) -> str:
        '''
        Shared method to save files to disk in Keras HDF5 format
        Then upload HDF5 file from disk to onedrive
        '''
        filename = f'{persistable_id}.h5'
        bucket = 'hdf5'
        cls.save_keras_object(obj, filename)
        cls.upload_to_onedrive(bucket, filename)
        return filename

    @classmethod
    def _load_keras_hdf5_from_onedrive(cls, filename: str, **kwargs) -> Any:
        '''
        Download HDF5 file from onedrive to disk
        Then load files from disk in HDF5 format
        '''
        bucket = 'hdf5'
        cls.download_from_onedrive(bucket, filename)
        return cls.load_keras_object(filename)


class CloudBase(ExternalArtifactsMixin):
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


@ExternalArtifactsMixin.Decorators.register_save_pattern(
    save_method='_save_pickle_to_cloud', load_method='_load_pickle_from_cloud')
class CloudPickleSaveMixin(CloudBase):
    '''
    Mixin class to save objects to Cloud in pickled format
    '''
    SAVE_PATTERN = 'cloud_pickled'

    @classmethod
    def _save_pickle_to_cloud(cls, obj: Any, persistable_id: str, **kwargs) -> str:
        '''
        Shared method to save files to disk in pickled format
        Then upload pickled file from disk to cloud
        '''
        filename = f'{persistable_id}.pkl'
        folder = 'pickle'
        cls.pickle_object(obj, filename)
        cls.upload_to_cloud(folder, filename)
        return filename

    @classmethod
    def _load_pickle_from_cloud(cls, filename: str, **kwargs) -> Any:
        '''
        Download pickled file from cloud to disk
        Then load files from disk in pickled format
        '''
        folder = 'pickle'
        cls.download_from_cloud(folder, filename)
        return cls.load_pickled_object(filename)


@ExternalArtifactsMixin.Decorators.register_save_pattern(
    save_method='_save_hdf5_to_cloud', load_method='_load_hdf5_from_cloud')
class CloudHDF5SaveMixin(CloudBase):
    '''
    Mixin class to save objects to Cloud in HDF5 format
    '''
    SAVE_PATTERN = 'cloud_hdf5'

    @classmethod
    def _save_hdf5_to_cloud(cls, obj: Any, persistable_id: str, **kwargs) -> str:
        '''
        Shared method to save files to disk in HDF5 format
        Then upload HDF5 file from disk to cloud
        '''
        filename = f'{persistable_id}.h5'
        folder = 'hdf5'
        cls.hickle_object(obj, filename)
        cls.upload_to_cloud(folder, filename)
        return filename

    @classmethod
    def _load_hdf5_from_cloud(cls, filename: str, **kwargs) -> Any:
        '''
        Download HDF5 file from cloud to disk
        Then load files from disk in HDF5 format
        '''
        folder = 'hdf5'
        cls.download_from_cloud(folder, filename)
        return cls.load_hickled_object(filename)


@ExternalArtifactsMixin.Decorators.register_save_pattern(
    save_method='_save_keras_hdf5_to_cloud', load_method='_load_keras_hdf5_from_cloud')
class CloudKerasHDF5SaveMixin(CloudBase):
    '''
    Mixin class to save objects to Cloud in Keras HDF5 format
    '''
    SAVE_PATTERN = 'cloud_keras_hdf5'

    @classmethod
    def _save_keras_hdf5_to_cloud(cls, obj: Any, persistable_id: str, **kwargs) -> str:
        '''
        Shared method to save files to disk in Keras HDF5 format
        Then upload HDF5 file from disk to cloud
        '''
        filename = f'{persistable_id}.h5'
        folder = 'hdf5'
        cls.save_keras_object(obj, filename)
        cls.upload_to_cloud(folder, filename)
        return filename

    @classmethod
    def _load_keras_hdf5_from_cloud(cls, filename: str, **kwargs) -> Any:
        '''
        Download HDF5 file from cloud to disk
        Then load files from disk in HDF5 format
        '''
        folder = 'hdf5'
        cls.download_from_cloud(folder, filename)
        return cls.load_keras_object(filename)


@ExternalArtifactsMixin.Decorators.register_save_pattern(
    save_pattern='cloud_keras_hdf5', save_method='_save_keras_hdf5_to_cloud', load_method='_load_keras_hdf5_from_cloud')
@ExternalArtifactsMixin.Decorators.register_save_pattern(
    save_pattern='cloud_hdf5', save_method='_save_hdf5_to_cloud', load_method='_load_hdf5_from_cloud')
@ExternalArtifactsMixin.Decorators.register_save_pattern(
    save_pattern='cloud_pickled', save_method='_save_pickle_to_cloud', load_method='_load_pickle_from_cloud')
@ExternalArtifactsMixin.Decorators.register_save_pattern(
    save_pattern='onedrive_keras_hdf5', save_method='_save_keras_hdf5_to_onedrive', load_method='_load_keras_hdf5_from_onedrive')
@ExternalArtifactsMixin.Decorators.register_save_pattern(
    save_pattern='onedrive_hdf5', save_method='_save_hdf5_to_onedrive', load_method='_load_hdf5_from_onedrive')
@ExternalArtifactsMixin.Decorators.register_save_pattern(
    save_pattern='onedrive_pickled', save_method='_save_pickle_to_onedrive', load_method='_load_pickle_from_onedrive')
@ExternalArtifactsMixin.Decorators.register_save_pattern(
    save_pattern='disk_keras_hdf5', save_method='_save_keras_hdf5_to_disk', load_method='_load_keras_hdf5_from_disk')
@ExternalArtifactsMixin.Decorators.register_save_pattern(
    save_pattern='disk_hdf5', save_method='_save_hdf5_to_disk', load_method='_load_hdf5_from_disk')
@ExternalArtifactsMixin.Decorators.register_save_pattern(
    save_pattern='disk_pickled', save_method='_save_pickle_to_disk', load_method='_load_pickle_from_disk')
@ExternalArtifactsMixin.Decorators.register_save_pattern(
    save_pattern='database_pickled', save_method='_save_pickle_to_database', load_method='_load_pickle_from_database')
@ExternalArtifactsMixin.Decorators.register_save_pattern(
    save_pattern='database_table', save_method='_save_dataframe_to_table', load_method='_load_dataframe_from_table')
class AllSaveMixin(DatabaseTableSaveMixin, DatabasePickleSaveMixin,
                   DiskPickleSaveMixin, DiskHDF5SaveMixin, KerasDiskHDF5SaveMixin,
                   OnedrivePickleSaveMixin, OnedriveHDF5SaveMixin, OnedriveKerasHDF5SaveMixin,
                   CloudPickleSaveMixin, CloudHDF5SaveMixin, CloudKerasHDF5SaveMixin):
    '''
    Convenience container to assemble all the save patterns into a single class
    '''
    SAVE_PATTERN = None
    SAVE_METHODS = {}
    LOAD_METHODS = {}
