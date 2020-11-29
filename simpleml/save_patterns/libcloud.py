'''
Module for cloud save pattern definitions
Uses Apache Libcloud as a universal engine
'''

__author__ = 'Elisha Yadgaran'


from os.path import join, isfile
from typing import Any

from simpleml.save_patterns.base import BaseSavePattern
from simpleml.save_patterns.decorators import SavePatternDecorators
from simpleml.utils.configuration import PICKLED_FILESTORE_DIRECTORY,\
    HDF5_FILESTORE_DIRECTORY, PICKLE_DIRECTORY, HDF5_DIRECTORY, CONFIG, CLOUD_SECTION
from simpleml.imports import Provider
from simpleml.imports import get_driver


class CloudSavePatternMixin(object):
    '''
    Mixin class to save/load objects via Apache Libcloud

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

    # Global initialization of cloud provider. Avoids reauthentication per persistable
    CLOUD_DRIVER = None

    @property
    def driver(self):
        '''
        "classproperty" to return and optionally globally set the cloud provider
        '''
        if self.__class__.CLOUD_DRIVER is None:
            cloud_section = CONFIG.get(CLOUD_SECTION, 'section')
            connection_params = CONFIG.getlist(cloud_section, 'connection_params')

            driver_cls = get_driver(getattr(Provider, CONFIG.get(cloud_section, 'driver')))
            driver = driver_cls(**{param: CONFIG.get(cloud_section, param) for param in connection_params})

            self.__class__.CLOUD_DRIVER = driver
        return self.__class__.CLOUD_DRIVER

    @classmethod
    def reset_driver(cls):
        '''
        Convenience method to set parsed driver back to None. Forces a config reread on
        next invocation
        '''
        cls.CLOUD_DRIVER = None

    def upload_to_cloud(self,
                        folder: str,
                        filename: str) -> None:
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

    def download_from_cloud(self,
                            folder: str,
                            filename: str) -> None:
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


class CloudBase(BaseSavePattern, CloudSavePatternMixin):
    '''
    Implementation class for extended base save patterns
    to/from the cloud via Apache Libcloud
    '''


@SavePatternDecorators.register_save_pattern
class CloudPickleSavePattern(CloudBase):
    '''
    Save pattern implementation to save objects to Cloud in pickled format
    '''
    SAVE_PATTERN = 'cloud_pickled'

    @classmethod
    def save(cls,
             obj: Any,
             persistable_id: str,
             **kwargs) -> str:
        '''
        Save method to save files to disk in pickled format
        Then upload pickled file from disk to cloud
        '''
        filename = f'{persistable_id}.pkl'
        folder = 'pickle'
        cls.pickle_object(obj, filename)
        cls.upload_to_cloud(folder, filename)
        return filename

    @classmethod
    def load(cls,
             filename: str,
             **kwargs) -> Any:
        '''
        Download pickled file from cloud to disk
        Then load files from disk in pickled format
        '''
        folder = 'pickle'
        cls.download_from_cloud(folder, filename)
        return cls.load_pickled_object(filename)


@SavePatternDecorators.register_save_pattern
class CloudHDF5SavePattern(CloudBase):
    '''
    Save pattern implementation to save objects to Cloud in HDF5 format
    '''
    SAVE_PATTERN = 'cloud_hdf5'

    @classmethod
    def save(cls,
             obj: Any,
             persistable_id: str,
             **kwargs) -> str:
        '''
        Save method to save files to disk in HDF5 format
        Then upload HDF5 file from disk to cloud
        '''
        filename = f'{persistable_id}.h5'
        folder = 'hdf5'
        cls.hickle_object(obj, filename)
        cls.upload_to_cloud(folder, filename)
        return filename

    @classmethod
    def load(cls,
             filename: str,
             **kwargs) -> Any:
        '''
        Download HDF5 file from cloud to disk
        Then load files from disk in HDF5 format
        '''
        folder = 'hdf5'
        cls.download_from_cloud(folder, filename)
        return cls.load_hickled_object(filename)


@SavePatternDecorators.register_save_pattern
class CloudKerasHDF5SavePattern(CloudBase):
    '''
    Save pattern implementation to save objects to Cloud in Keras HDF5 format
    '''
    SAVE_PATTERN = 'cloud_keras_hdf5'

    @classmethod
    def save(cls,
             obj: Any,
             persistable_id: str,
             **kwargs) -> str:
        '''
        Save method to save files to disk in Keras HDF5 format
        Then upload HDF5 file from disk to cloud
        '''
        filename = f'{persistable_id}.h5'
        folder = 'hdf5'
        cls.save_keras_object(obj, filename)
        cls.upload_to_cloud(folder, filename)
        return filename

    @classmethod
    def load(cls,
             filename: str,
             **kwargs) -> Any:
        '''
        Download HDF5 file from cloud to disk
        Then load files from disk in HDF5 format
        '''
        folder = 'hdf5'
        cls.download_from_cloud(folder, filename)
        return cls.load_keras_object(filename)
