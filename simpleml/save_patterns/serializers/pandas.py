'''
Pandas Save/Load Utils and Patterns
'''

__author__ = 'Elisha Yadgaran'


from io import StringIO
from os.path import isfile, join
from typing import Any, Dict, Optional

import pandas as pd
from simpleml.registries import FILEPATH_REGISTRY
from simpleml.save_patterns.base import BaseSerializer
from simpleml.utils.configuration import (CSV_DIRECTORY, JSON_DIRECTORY,
                                          PARQUET_DIRECTORY)


class PandasPersistenceMethods(object):
    '''
    Base class for internal Pandas serialization/deserialization options

    Wraps pd.Dataframe methods with sensible defaults

    https://pandas.pydata.org/docs/reference/io.html
    https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html
    '''
    INDEX_COLUMN = 'simpleml_index'

    @classmethod
    def read_csv(cls, filename: str, **kwargs) -> pd.DataFrame:
        '''Helper method to read in a csv file'''
        df = pd.read_csv(filename, **kwargs)
        if cls.INDEX_COLUMN in df.columns:
            df = df.set_index(cls.INDEX_COLUMN)
        return df

    @staticmethod
    def read_parquet(filepath: str,
                     **kwargs) -> pd.DataFrame:
        return pd.read_parquet(filepath, **kwargs)

    @staticmethod
    def read_hdf(filepath: str,
                 **kwargs) -> pd.DataFrame:
        return pd.read_hdf(filepath, **kwargs)

    @staticmethod
    def read_orc(filepath: str,
                 **kwargs) -> pd.DataFrame:
        return pd.read_orc(filepath, **kwargs)

    @classmethod
    def read_json(cls,
                  filepath: str,
                  orient: str = 'records',
                  lines: bool = True,
                  **kwargs) -> pd.DataFrame:
        # Automatically handle index
        df = pd.read_json(filepath, orient=orient, lines=lines, **kwargs)
        if cls.INDEX_COLUMN in df.columns:
            df = df.set_index(cls.INDEX_COLUMN)
        return df

    @staticmethod
    def read_fwf(**kwargs) -> pd.DataFrame:
        return pd.read_fwf(**kwargs)

    @staticmethod
    def read_html(**kwargs) -> pd.DataFrame:
        return pd.read_html(**kwargs)

    @staticmethod
    def read_xml(**kwargs) -> pd.DataFrame:
        return pd.read_xml(**kwargs)

    @staticmethod
    def read_clipboard(**kwargs) -> pd.DataFrame:
        return pd.read_clipboard(**kwargs)

    @staticmethod
    def read_excel(**kwargs) -> pd.DataFrame:
        return pd.read_excel(**kwargs)

    @staticmethod
    def read_feather(**kwargs) -> pd.DataFrame:
        return pd.read_feather(**kwargs)

    @staticmethod
    def read_stata(**kwargs) -> pd.DataFrame:
        return pd.read_stata(**kwargs)

    @staticmethod
    def read_sas(**kwargs) -> pd.DataFrame:
        return pd.read_sas(**kwargs)

    @staticmethod
    def read_spss(**kwargs) -> pd.DataFrame:
        return pd.read_spss(**kwargs)

    @staticmethod
    def read_pickle(**kwargs) -> pd.DataFrame:
        return pd.read_pickle(**kwargs)

    @staticmethod
    def read_sql(**kwargs) -> pd.DataFrame:
        return pd.read_sql(**kwargs)

    @staticmethod
    def read_bigquery(**kwargs) -> pd.DataFrame:
        return pd.read_gbq(**kwargs)

    @staticmethod
    def read_sql_table(**kwargs) -> pd.DataFrame:
        return pd.read_sql_table(**kwargs)

    @staticmethod
    def read_table(**kwargs) -> pd.DataFrame:
        return pd.read_table(**kwargs)

    @staticmethod
    def read_sql_query(query: str,
                       connection,
                       **kwargs) -> pd.DataFrame:
        '''Helper method to read in sql data'''
        return pd.read_sql_query(query, connection, **kwargs)

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
    def to_pickle(df: pd.DataFrame,
                  filepath: str,
                  overwrite: bool = True,
                  **kwargs) -> None:
        if not overwrite:
            # Check if file was already serialized
            if isfile(filepath):
                return
        df.to_pickle(filepath, **kwargs)

    @classmethod
    def to_csv(cls, df: pd.DataFrame,
               filepath: str,
               overwrite: bool = True,
               **kwargs) -> None:
        if not overwrite:
            # Check if file was already serialized
            if isfile(filepath):
                return
        df.to_csv(filepath, index_label=cls.INDEX_COLUMN, **kwargs)

    @staticmethod
    def to_clipboard(df: pd.DataFrame,
                     filepath: str,
                     overwrite: bool = True,
                     **kwargs) -> None:
        if not overwrite:
            # Check if file was already serialized
            if isfile(filepath):
                return
        df.to_clipboard(filepath, **kwargs)

    @staticmethod
    def to_excel(df: pd.DataFrame,
                 filepath: str,
                 overwrite: bool = True,
                 **kwargs) -> None:
        if not overwrite:
            # Check if file was already serialized
            if isfile(filepath):
                return
        df.to_excel(filepath, **kwargs)

    @classmethod
    def to_json(cls, df: pd.DataFrame,
                filepath: str,
                overwrite: bool = True,
                lines: bool = True,
                orient: str = 'records',
                **kwargs) -> None:
        if not overwrite:
            # Check if file was already serialized
            if isfile(filepath):
                return
        # json records do not include index so artificially inject
        if cls.INDEX_COLUMN in df.columns:
            df.to_json(filepath, orient=orient, lines=lines ** kwargs)
        else:
            df.reset_index(drop=False).rename(
                columns={'index': cls.INDEX_COLUMN}
            ).to_json(filepath, orient=orient, lines=lines, **kwargs)

    @staticmethod
    def to_html(df: pd.DataFrame,
                filepath: str,
                overwrite: bool = True,
                **kwargs) -> None:
        if not overwrite:
            # Check if file was already serialized
            if isfile(filepath):
                return
        df.to_html(filepath, **kwargs)

    @staticmethod
    def to_xml(df: pd.DataFrame,
               filepath: str,
               overwrite: bool = True,
               **kwargs) -> None:
        if not overwrite:
            # Check if file was already serialized
            if isfile(filepath):
                return
        df.to_xml(filepath, **kwargs)

    @staticmethod
    def to_latex(df: pd.DataFrame,
                 filepath: str,
                 overwrite: bool = True,
                 **kwargs) -> None:
        if not overwrite:
            # Check if file was already serialized
            if isfile(filepath):
                return
        df.to_latex(filepath, **kwargs)

    @staticmethod
    def to_feather(df: pd.DataFrame,
                   filepath: str,
                   overwrite: bool = True,
                   **kwargs) -> None:
        if not overwrite:
            # Check if file was already serialized
            if isfile(filepath):
                return
        df.to_feather(filepath, **kwargs)

    @staticmethod
    def to_parquet(df: pd.DataFrame,
                   filepath: str,
                   overwrite: bool = True,
                   **kwargs) -> None:
        if not overwrite:
            # Check if file was already serialized
            if isfile(filepath):
                return
        df.to_parquet(filepath, **kwargs)

    @staticmethod
    def to_stata(df: pd.DataFrame,
                 filepath: str,
                 overwrite: bool = True,
                 **kwargs) -> None:
        if not overwrite:
            # Check if file was already serialized
            if isfile(filepath):
                return
        df.to_stata(filepath, **kwargs)


class PandasParquetSerializer(BaseSerializer):
    @staticmethod
    def serialize(obj: pd.DataFrame,
                  filepath: str,
                  format_directory: str = PARQUET_DIRECTORY,
                  format_extension: str = '.parquet',
                  destination_directory: str = 'system_temp',
                  **kwargs) -> Dict[str, str]:

        # Append the filepath to the storage directory
        filepath = join(format_directory, filepath + format_extension)
        full_path = join(FILEPATH_REGISTRY.get(destination_directory), filepath)
        PandasPersistenceMethods.to_parquet(obj, full_path)
        return {'filepath': filepath, 'source_directory': destination_directory}

    @staticmethod
    def deserialize(filepath: str,
                    source_directory: str = 'system_temp',
                    **kwargs) -> Dict[str, pd.DataFrame]:
        full_path = join(FILEPATH_REGISTRY.get(source_directory), filepath)
        return {'obj': PandasPersistenceMethods.read_parquet(full_path)}


class PandasCSVSerializer(BaseSerializer):
    @staticmethod
    def serialize(obj: pd.DataFrame,
                  filepath: str,
                  format_directory: str = CSV_DIRECTORY,
                  format_extension: str = '.csv',
                  destination_directory: str = 'system_temp',
                  **kwargs) -> Dict[str, str]:

        # Append the filepath to the storage directory
        filepath = join(format_directory, filepath + format_extension)
        full_path = join(FILEPATH_REGISTRY.get(destination_directory), filepath)
        PandasPersistenceMethods.to_csv(obj, full_path)
        return {'filepath': filepath, 'source_directory': destination_directory}

    @staticmethod
    def deserialize(filepath: str,
                    source_directory: str = 'system_temp',
                    **kwargs) -> Dict[str, pd.DataFrame]:
        full_path = join(FILEPATH_REGISTRY.get(source_directory), filepath)
        return {'obj': PandasPersistenceMethods.read_csv(full_path)}


class PandasJSONSerializer(BaseSerializer):
    @staticmethod
    def serialize(obj: pd.DataFrame,
                  filepath: str,
                  format_directory: str = JSON_DIRECTORY,
                  format_extension: str = '.jsonl',
                  destination_directory: str = 'system_temp',
                  **kwargs) -> Dict[str, str]:

        # Append the filepath to the storage directory
        filepath = join(format_directory, filepath + format_extension)
        full_path = join(FILEPATH_REGISTRY.get(destination_directory), filepath)
        PandasPersistenceMethods.to_json(obj, full_path)
        return {'filepath': filepath, 'source_directory': destination_directory}

    @staticmethod
    def deserialize(filepath: str,
                    source_directory: str = 'system_temp',
                    **kwargs) -> Dict[str, pd.DataFrame]:
        full_path = join(FILEPATH_REGISTRY.get(source_directory), filepath)
        return {'obj': PandasPersistenceMethods.read_json(full_path)}
