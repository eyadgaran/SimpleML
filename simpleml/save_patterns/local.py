'''
Module for save patterns registered for local persistence
'''

__author__ = 'Elisha Yadgaran'


from typing import Any

from simpleml.save_patterns.decorators import SavePatternDecorators
from simpleml.save_patterns.base import BaseSavePattern


@SavePatternDecorators.register_save_pattern
class DiskPickleSavePattern(BaseSavePattern):
    '''
    Save pattern implementation to save objects to disk in pickled format
    '''
    SAVE_PATTERN = 'disk_pickled'

    @classmethod
    def save(cls, obj: Any, persistable_id: str, **kwargs) -> str:
        '''
        Save method to save files to disk in pickled format
        '''
        filename = f'{persistable_id}.pkl'
        cls.pickle_object(obj, filename)
        return filename

    @classmethod
    def load(cls, filename: str, **kwargs) -> Any:
        '''
        Load method to load files from disk in pickled format
        '''
        return cls.load_pickled_object(filename)


@SavePatternDecorators.register_save_pattern
class DiskHDF5SavePattern(BaseSavePattern):
    '''
    Save pattern implementation to save objects to disk in HDF5 format with hickle
    '''
    SAVE_PATTERN = 'disk_hdf5'

    @classmethod
    def save(cls, obj: Any, persistable_id: str, **kwargs) -> str:
        '''
        Save method to save files to disk in hickle's HDF5 format
        '''
        filename = f'{persistable_id}.h5'
        cls.hickle_object(obj, filename)
        return filename

    @classmethod
    def load(cls, filename: str, **kwargs) -> Any:
        '''
        Load method to load files from disk in hickle's HDF5 format
        '''
        return cls.load_hickled_object(filename)


@SavePatternDecorators.register_save_pattern
class KerasDiskHDF5SavePattern(BaseSavePattern):
    '''
    Save pattern implementation to save objects to disk in Keras's HDF5 format
    Keras's internal persistence mechanism utilizes HDF5 and implements a custom pattern
    '''
    SAVE_PATTERN = 'disk_keras_hdf5'

    @classmethod
    def save(cls, obj: Any, persistable_id: str, **kwargs) -> str:
        '''
        Save method to save files to disk in Keras's HDF5 format
        '''
        filename = f'{persistable_id}.h5'
        cls.save_keras_object(obj, filename)
        return filename

    @classmethod
    def load(cls, filename: str, **kwargs) -> Any:
        '''
        Load method to load files from disk in Keras's HDF5 format
        '''
        return cls.load_keras_object(filename)
