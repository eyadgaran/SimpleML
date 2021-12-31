'''
Module for Hickle save patterns
'''

__author__ = 'Elisha Yadgaran'


from os.path import isfile, join
from typing import Any, Dict

from simpleml.imports import hickle
from simpleml.save_patterns.base import BaseSerializer
from simpleml.utils.configuration import HDF5_DIRECTORY


class HicklePersistenceMethods(object):
    '''
    Base class for Hickle serialization/deserialization options
    '''

    @staticmethod
    def dump_object(obj: Any,
                    filepath: str,
                    overwrite: bool = True,
                    **kwargs) -> None:
        '''
        Serializes an object to the filesystem in HDF5 format.

        :param overwrite: Boolean indicating whether to first check if HDF5
            object is already serialized. Defaults to not checking, but can be
            leverage by implementations that want the same artifact in multiple
            places
        '''
        if not overwrite:
            # Check if file was already serialized
            if isfile(filepath):
                return

        # Defaults
        if 'compression' not in kwargs:
            kwargs['compression'] = 'gzip'
        if 'compression_opts' not in kwargs:
            kwargs['compression_opts'] = 9

        hickle.dump(obj, filepath, **kwargs)

    @staticmethod
    def load_object(filepath: str, **kwargs) -> Any:
        '''
        Loads an object from the filesystem.
        '''
        return hickle.load(filepath, **kwargs)


class CloudpickleFileSerializer(BaseSerializer):
    @staticmethod
    def serialize(obj: Any,
                  filepath: str,
                  format_directory: str = HDF5_DIRECTORY,
                  format_extension: str = '.hdf5',
                  destination_directory: str = 'system_temp',
                  **kwargs) -> Dict[str, str]:

        # Append the filepath to the pickle storage directory
        filepath = join(format_directory, filepath + format_extension)
        full_path = join(FILEPATH_REGISTRY.get(destination_directory), filepath)
        # make sure the directory exists
        makedirs(dirname(full_path), exist_ok=True)

        HicklePersistenceMethods.dump_object(obj, full_path)

        return {'filepath': filepath, 'source_directory': destination_directory}

    @staticmethod
    def deserialize(filepath: str,
                    source_directory: str = 'system_temp', **kwargs) -> Dict[str, Any]:
        full_path = join(FILEPATH_REGISTRY.get(source_directory), filepath)

        return {'obj': HicklePersistenceMethods.load_object(full_path)}
