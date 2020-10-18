'''
Module for base save pattern definition
'''

__author__ = 'Elisha Yadgaran'


import pandas as pd
import cloudpickle as pickle

from os.path import join, isfile
from typing import Optional, Any, Union, Dict
from io import StringIO
from abc import abstractmethod, ABCMeta

from simpleml.utils.configuration import PICKLED_FILESTORE_DIRECTORY,\
    HDF5_FILESTORE_DIRECTORY
from simpleml.registries import KERAS_REGISTRY
# Import optional dependencies
from simpleml.imports import load_model, hickle


class SavePatternMixin(object):
    '''
    Mixin class with methods for different save operations
    '''
    @staticmethod
    def df_to_sql(engine,
                  df: pd.DataFrame,
                  table: str,
                  dtype: Optional[Dict[str, str]] = None,
                  schema: str = 'public',
                  if_exists: str = 'replace',
                  sep: str = '|',
                  encoding: str = 'utf8',
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
    def pickle_object(obj: Any,
                      filepath: Optional[str] = None,
                      overwrite: bool = True,
                      root_directory: str = PICKLED_FILESTORE_DIRECTORY) -> Union[str, None]:
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
        filepath = join(root_directory, filepath)

        if not overwrite:
            # Check if file was already serialized
            if isfile(filepath):
                return

        with open(filepath, 'wb') as pickled_file:
            pickle.dump(obj, pickled_file)  # , protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_pickled_object(filepath: str,
                            stream: bool = False,
                            root_directory: str = PICKLED_FILESTORE_DIRECTORY) -> Any:
        '''
        Loads an object from a serialized string or filesystem. When stream is
        True, it tries to load the file directly from the string.

        Prepends path to SimpleML Pickle directory before loading. ONLY pass in
        a relative filepath from that location
        '''
        if stream:
            return pickle.loads(filepath)

        with open(join(root_directory, filepath), 'rb') as pickled_file:
            return pickle.load(pickled_file)

    @staticmethod
    def hickle_object(obj: Any,
                      filepath: str,
                      overwrite: bool = True,
                      root_directory: str = HDF5_FILESTORE_DIRECTORY) -> None:
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
        hickle_file = join(root_directory, filepath)
        if not overwrite:
            # Check if file was already serialized
            if isfile(hickle_file):
                return
        hickle.dump(obj, hickle_file, compression='gzip', compression_opts=9)

    @staticmethod
    def load_hickled_object(filepath: str,
                            root_directory: str = HDF5_FILESTORE_DIRECTORY) -> Any:
        '''
        Loads an object from the filesystem.

        Prepends path to SimpleML HDF5 directory before loading. ONLY pass in
        a relative filepath from that location
        '''
        hickle_file = join(root_directory, filepath)
        return hickle.load(hickle_file)

    @staticmethod
    def save_keras_object(obj: Any,
                          filepath: str,
                          overwrite: bool = True,
                          root_directory: str = HDF5_FILESTORE_DIRECTORY) -> None:
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
        hdf5_file = join(root_directory, filepath)
        if not overwrite:
            # Check if file was already serialized
            if isfile(hdf5_file):
                return
        obj.save(hdf5_file)

    @staticmethod
    def load_keras_object(filepath: str,
                          root_directory: str = HDF5_FILESTORE_DIRECTORY) -> Any:
        '''
        Loads a Keras object from the filesystem.

        Prepends path to SimpleML HDF5 directory before loading. ONLY pass in
        a relative filepath from that location
        '''
        return load_model(
            str(join(root_directory, filepath)),
            custom_objects=KERAS_REGISTRY.registry)

    @staticmethod
    def load_sql(query: str,
                 connection,
                 **kwargs) -> pd.DataFrame:
        '''Helper method to read in sql data'''
        return pd.read_sql_query(query, connection, **kwargs)


class BaseSavePattern(SavePatternMixin, metaclass=ABCMeta):
    '''
    Abstract base class for save patterns
    '''
    @abstractmethod
    def save(self):
        '''
        The save method invoked
        '''

    @abstractmethod
    def load(self):
        '''
        The load method invoked
        '''
