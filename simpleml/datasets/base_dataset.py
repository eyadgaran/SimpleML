from simpleml.persistables.base_persistable import BasePersistable
from abc import abstractmethod
import pandas as pd
from pandas.util import hash_pandas_object
import cStringIO


__author__ = 'Elisha Yadgaran'


class BaseDataset(BasePersistable):
    '''
    Base class for all Dataset objects.

    Every dataset has one dataframe associated with it and can be subdivided
    by inheriting classes (y column for supervised, train/test/validation splits, etc)

    Dataset storage is the final resulting dataframe so technically a dataset
    is uniquely determined by Dataset class + Dataset Pipeline

    -------
    Schema
    -------
    No additional columns
    '''

    __abstract__ = True

    def __init__(self, has_external_files=True, *args, **kwargs):
        super(BaseDataset, self).__init__(
            has_external_files=has_external_files, *args, **kwargs)

        # Instantiate dataframe variable - doesn't get populated until
        # build_dataframe() is called
        self._dataframe = None

    @property
    def dataframe(self):
        # Return dataframe if generated, otherwise generate first
        if self._dataframe is None:
            self.build_dataframe()
        return self._dataframe

    @abstractmethod
    def build_dataframe(self):
        '''
        Must set self._dataframe
        '''

    def _hash(self):
        '''
        Datasets rely on external data so instead of hashing the config,
        hash the actual resulting dataframe
        This requires loading the data before determining duplication
        so overwrite for differing behavior
        '''
        return hash_pandas_object(self.dataframe, index=False).sum()

    def _save_external_files(self, schema, engine):
        '''
        Shared method to save dataframe into a new table with name = GUID

        Hardcoded to only store in database so overwrite to use pickled
        objects or other storage mechanism
        '''
        self.filepaths = {"database": [(schema, str(self.id))]}
        self.df_to_sql(engine, self.dataframe,
                       str(self.id), schema=schema)

    def _load_external_files(self, engine):
        '''
        Shared method to load dataframe from database

        Hardcoded to only pull from database so overwrite to use pickled
        objects or other storage mechanism
        '''
        schema, tablename = self.filepaths['database'][0]
        self._dataframe = self.load_sql(
            'select * from "{}"."{}"'.format(schema, tablename),
            engine
        )

    @staticmethod
    def load_csv(filename, **kwargs):
        '''Helper method to read in a csv file'''
        return pd.read_csv(filename, **kwargs)

    @staticmethod
    def load_sql(query, connection, **kwargs):
        '''Helper method to read in sql data'''
        return pd.read_sql_query(query, connection, **kwargs)

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

        # Create Table
        df.head(0).to_sql(table, con=engine, if_exists=if_exists,
                          index=index, schema=schema, dtype=dtype)

        # Prepare data
        output = cStringIO.StringIO()
        df.to_csv(output, sep=sep, header=False, encoding=encoding, index=index)
        output.seek(0)

        # Insert data
        connection = engine.raw_connection()
        cursor = connection.cursor()
        cursor.copy_from(output, '"' + '"."'.join([schema, table]) + '"', sep=sep, null='',
                         columns=['"{}"'.format(i) for i in df.columns])
        connection.commit()
        connection.close()
