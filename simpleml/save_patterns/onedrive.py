'''
Module for Microsoft Onedrive save pattern definitions
'''

__author__ = 'Elisha Yadgaran'


from os.path import join, isfile
from typing import Any

from simpleml.imports import onedrivesdk
from simpleml.save_patterns.base import BaseSavePattern
from simpleml.save_patterns.decorators import SavePatternDecorators
from simpleml.utils.configuration import PICKLED_FILESTORE_DIRECTORY,\
    HDF5_FILESTORE_DIRECTORY, PICKLE_DIRECTORY, HDF5_DIRECTORY, CONFIG


class OnedriveBase(BaseSavePattern):
    '''
    Base class to save/load objects to Microsoft Onedrive
    '''
    # Shared Connections
    ONEDRIVE_SECTION = 'onedrive'
    ONEDRIVE_CLIENT = None
    ONEDRIVE_ROOT_ID = None
    ONEDRIVE_FILESTORE_ID = None
    ONEDRIVE_PICKLE_ID = None
    ONEDRIVE_HDF5_ID = None

    @property
    def client(self):
        if self.__class__.ONEDRIVE_CLIENT is None:
            self.authenticate_onedrive()
        try:
            # See if existing client is configured with a valid token
            from onedrivesdk.error import OneDriveError
            self.__class__.ONEDRIVE_CLIENT.item(id='root').get()
        except OneDriveError:
            self.authenticate_onedrive()
        return self.__class__.ONEDRIVE_CLIENT

    @client.setter
    def client(self, value):
        self.__class__.ONEDRIVE_CLIENT = value

    @property
    def onedrive_root_id(self):
        if self.__class__.ONEDRIVE_ROOT_ID is None:
            self.create_onedrive_schema()
        return self.__class__.ONEDRIVE_ROOT_ID

    @onedrive_root_id.setter
    def onedrive_root_id(self, value):
        self.__class__.ONEDRIVE_ROOT_ID = value

    @property
    def onedrive_filestore_id(self):
        if self.__class__.ONEDRIVE_FILESTORE_ID is None:
            self.create_onedrive_schema()
        return self.__class__.ONEDRIVE_FILESTORE_ID

    @onedrive_filestore_id.setter
    def onedrive_filestore_id(self, value):
        self.__class__.ONEDRIVE_FILESTORE_ID = value

    @property
    def onedrive_pickle_id(self):
        if self.__class__.ONEDRIVE_PICKLE_ID is None:
            self.create_onedrive_schema()
        return self.__class__.ONEDRIVE_PICKLE_ID

    @onedrive_pickle_id.setter
    def onedrive_pickle_id(self, value):
        self.__class__.ONEDRIVE_PICKLE_ID = value

    @property
    def onedrive_hdf5_id(self):
        if self.__class__.ONEDRIVE_HDF5_ID is None:
            self.create_onedrive_schema()
        return self.__class__.ONEDRIVE_HDF5_ID

    @onedrive_hdf5_id.setter
    def onedrive_hdf5_id(self, value):
        self.__class__.ONEDRIVE_HDF5_ID = value

    def authenticate_onedrive(self):
        '''
        Authenticate with Onedrive Oauth2
        '''
        from onedrivesdk.helpers.GetAuthCodeServer import get_auth_code

        section = CONFIG[self.ONEDRIVE_SECTION]
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


@SavePatternDecorators.register_save_pattern
class OnedrivePickleSavePattern(OnedriveBase):
    '''
    Save pattern implementation to save objects to Microsoft Onedrive in pickled format
    '''
    SAVE_PATTERN = 'onedrive_pickled'

    @classmethod
    def save(cls,
             obj: Any,
             persistable_id: str,
             **kwargs) -> str:
        '''
        Save method to save files to disk in pickled format
        Then upload pickled file from disk to onedrive
        '''
        filename = f'{persistable_id}.pkl'
        bucket = 'pickle'
        cls.pickle_object(obj, filename)
        cls.upload_to_onedrive(bucket, filename)
        return filename

    @classmethod
    def load(cls,
             filename: str,
             **kwargs) -> Any:
        '''
        Download pickled file from onedrive to disk
        Then load files from disk in pickled format
        '''
        bucket = 'pickle'
        cls.download_from_onedrive(bucket, filename)
        return cls.load_pickled_object(filename)


@SavePatternDecorators.register_save_pattern
class OnedriveHDF5SavePattern(OnedriveBase):
    '''
    Save pattern implementation to save objects to Microsoft Onedrive in HDF5 format
    '''
    SAVE_PATTERN = 'onedrive_hdf5'

    @classmethod
    def save(cls,
             obj: Any,
             persistable_id: str,
             **kwargs) -> str:
        '''
        Save method to save files to disk in HDF5 format
        Then upload HDF5 file from disk to onedrive
        '''
        filename = f'{persistable_id}.h5'
        bucket = 'hdf5'
        cls.hickle_object(obj, filename)
        cls.upload_to_onedrive(bucket, filename)
        return filename

    @classmethod
    def load(cls,
             filename: str,
             **kwargs) -> Any:
        '''
        Download HDF5 file from onedrive to disk
        Then load files from disk in HDF5 format
        '''
        bucket = 'hdf5'
        cls.download_from_onedrive(bucket, filename)
        return cls.load_hickled_object(filename)


@SavePatternDecorators.register_save_pattern
class OnedriveKerasHDF5SavePattern(OnedriveBase):
    '''
    Save pattern implementation to save objects to Microsoft Onedrive in Keras HDF5 format
    '''
    SAVE_PATTERN = 'onedrive_keras_hdf5'

    @classmethod
    def save(cls,
             obj: Any,
             persistable_id: str,
             **kwargs) -> str:
        '''
        Save method to save files to disk in Keras HDF5 format
        Then upload HDF5 file from disk to onedrive
        '''
        filename = f'{persistable_id}.h5'
        bucket = 'hdf5'
        cls.save_keras_object(obj, filename)
        cls.upload_to_onedrive(bucket, filename)
        return filename

    @classmethod
    def load(cls,
             filename: str,
             **kwargs) -> Any:
        '''
        Download HDF5 file from onedrive to disk
        Then load files from disk in HDF5 format
        '''
        bucket = 'hdf5'
        cls.download_from_onedrive(bucket, filename)
        return cls.load_keras_object(filename)
