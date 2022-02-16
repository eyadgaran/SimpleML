'''
Module to centralize all external imports - makes it easy to handle
optional dependencies in different installations
'''

__author__ = 'Elisha Yadgaran'


import logging

LOGGER = logging.getLogger(__name__)


# class MissingImportMetaclass(type):
#     '''
#     TODO: figure out how to dynamically resolve metaclasses with conflicting mixins
#
#     Separate metaclass to circumvent cyclical dependency between factory and import
#     wrapper
#     '''
#
#     def __getattr__(self, attr):
#         '''
#         Recursively return ImportWrappers until a method is actually called
#         '''
#         return MissingImportFactory(f'{self.name}.{attr}', self.pypi_name, self.simpleml_extra_group)


class MissingImportWrapper(object):
    '''
    Wrapper class and callable generator to be used instead of unavailable dependencies
    Errors on reference when not available instead of on import
    '''

    def __init__(self, *args, **kwargs):
        self.raise_error()

    def __new__(cls, *args, **kwargs):
        cls.raise_error()

    @classmethod
    def raise_error(cls):
        raise ImportError(f'Attempting to use missing dependency {cls.name}. Install via `pip install {cls.pypi_name}` or `pip install simpleml[{cls.simpleml_extra_group}]` and restart script')

    @classmethod
    def __call__(cls):
        cls.raise_error()

    @classmethod
    def __repr__(cls):
        return f'Missing Dependency Wrapper for {cls.name} (`pip install {cls.pypi_name}` or `pip install simpleml[{cls.simpleml_extra_group}]`)'


class MissingImportFactory(object):
    '''
    Factory class to generate library specific import wrappers
    '''
    def __new__(cls,
                name: str,
                pypi_name: str,
                simpleml_extra_group: str):
        LOGGER.debug(f'Wrapping missing dependency: {name}')

        class LibraryMissingImportWrapper(MissingImportWrapper):
            '''
            Wrap as a library specific instance
            '''
            pass
        LibraryMissingImportWrapper.name = name
        LibraryMissingImportWrapper.pypi_name = pypi_name
        LibraryMissingImportWrapper.simpleml_extra_group = simpleml_extra_group
        return LibraryMissingImportWrapper


# Import optional dependencies or set to wrapper to avoid import errors
try:
    import psycopg2
except ImportError:
    psycopg2 = MissingImportFactory('psycopg2', 'psycopg2', 'postgres')

try:
    import tensorflow as tf
    import tf.keras as keras
    from tf.keras.models import Sequential, Model, load_model
    from tf.keras.utils import Sequence
except ImportError:
    tf = MissingImportFactory('tensorflow', 'tensorflow', 'deep-learning')
    keras = MissingImportFactory('tensorflow.keras', 'tensorflow', 'deep-learning')
    load_model = MissingImportFactory('tensorflow.keras.models.load_model', 'tensorflow', 'deep-learning')
    Sequential = MissingImportFactory('tensorflow.keras.models.Sequential', 'tensorflow', 'deep-learning')
    Model = MissingImportFactory('tensorflow.keras.models.Model', 'tensorflow', 'deep-learning')
    Sequence = MissingImportFactory('tensorflow.keras.utils.Sequence', 'tensorflow', 'deep-learning')

try:
    import hickle
except ImportError:
    hickle = MissingImportFactory('hickle', 'hickle', 'deep-learning')

try:
    import onedrivesdk
except ImportError:
    onedrivesdk = MissingImportFactory('onedrivesdk', 'onedrivesdk<2', 'onedrive')

try:
    from sshtunnel import SSHTunnelForwarder
except ImportError:
    SSHTunnelForwarder = MissingImportFactory('sshtunnel.SSHTunnelForwarder', 'sshtunnel', 'cloud')

try:
    from libcloud.storage.types import Provider
    from libcloud.storage.providers import get_driver
except ImportError:
    Provider = MissingImportFactory('libcloud.storage.types.Provider', 'apache-libcloud', 'cloud')
    get_driver = MissingImportFactory('libcloud.storage.providers.get_driver', 'apache-libcloud', 'cloud')

try:
    import dask.dataframe as dd
    ddDataFrame = dd.DataFrame
    ddSeries = dd.Series
except ImportError:
    dd = MissingImportFactory('dask.dataframe', 'dask', 'dask')
    ddDataFrame = MissingImportFactory('dask.dataframe.DataFrame', 'dask', 'dask')
    ddSeries = MissingImportFactory('dask.dataframe.Series', 'dask', 'dask')
