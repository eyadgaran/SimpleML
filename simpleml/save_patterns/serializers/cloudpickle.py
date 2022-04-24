'''
Module for Cloudpickle save patterns
'''

__author__ = 'Elisha Yadgaran'


from os import makedirs
from os.path import dirname, isfile, join
from typing import Any, Dict, Optional

import cloudpickle as pickle

from simpleml.registries import FILEPATH_REGISTRY
from simpleml.save_patterns.base import BaseSerializer
from simpleml.utils.configuration import PICKLE_DIRECTORY


class CloudpicklePersistenceMethods(object):
    '''
    Base class for internal cloudpickle serialization/deserialization options
    '''
    @staticmethod
    def dump_object(obj: Any,
                    filepath: str,
                    overwrite: bool = True) -> None:
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
        if not overwrite:
            # Check if file was already serialized
            if isfile(filepath):
                return

        with open(filepath, 'wb') as pickled_file:
            pickle.dump(obj, pickled_file)  # , protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def dumps_object(obj: Any) -> str:
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
        return pickle.dumps(obj)  # , protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_object(filepath: str) -> Any:
        '''
        Loads an object from a serialized string or filesystem. When stream is
        True, it tries to load the file directly from the string.

        Prepends path to SimpleML Pickle directory before loading. ONLY pass in
        a relative filepath from that location
        '''
        with open(filepath, 'rb') as pickled_file:
            return pickle.load(pickled_file)

    @staticmethod
    def loads_object(data: str) -> Any:
        '''
        Loads an object from a serialized string or filesystem. When stream is
        True, it tries to load the file directly from the string.

        Prepends path to SimpleML Pickle directory before loading. ONLY pass in
        a relative filepath from that location
        '''
        return pickle.loads(data)


class CloudpickleFileSerializer(BaseSerializer):
    @staticmethod
    def serialize(obj: Any,
                  filepath: str,
                  format_directory: str = PICKLE_DIRECTORY,
                  format_extension: str = '.pkl',
                  destination_directory: str = 'system_temp',
                  **kwargs) -> Dict[str, str]:

        # Append the filepath to the pickle storage directory
        filepath = join(format_directory, filepath + format_extension)
        full_path = join(FILEPATH_REGISTRY.get(destination_directory), filepath)
        # make sure the directory exists
        makedirs(dirname(full_path), exist_ok=True)

        CloudpicklePersistenceMethods.dump_object(obj, full_path)

        return {'filepath': filepath, 'source_directory': destination_directory}

    @staticmethod
    def deserialize(filepath: str,
                    source_directory: str = 'system_temp', **kwargs) -> Dict[str, Any]:
        full_path = join(FILEPATH_REGISTRY.get(source_directory), filepath)

        return {'obj': CloudpicklePersistenceMethods.load_object(full_path)}


class CloudpickleInMemorySerializer(BaseSerializer):
    @staticmethod
    def serialize(obj: Any, **kwargs) -> Dict[str, str]:
        data = CloudpicklePersistenceMethods.dumps_object(obj)
        return {'obj': data}

    @staticmethod
    def deserialize(obj: str, **kwargs) -> Dict[str, Any]:
        return {'obj': CloudpicklePersistenceMethods.loads_object(obj)}
