from simpleml.persistables.base_persistable import BasePersistable
import pandas as pd
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

    def __init__(self, has_external_files=True, **kwargs):
        # By default assume unsupervised so no targets
        label_columns = kwargs.pop('label_columns', [])

        super(BaseDataset, self).__init__(
            has_external_files=has_external_files, **kwargs)

        self.config['label_columns'] = label_columns

        # Instantiate dataframe variable - doesn't get populated until
        # build_dataframe() is called
        self._dataframe = None

    @property
    def dataframe(self):
        # Return dataframe if generated, otherwise generate first
        if self.unloaded_externals:
            self._load_external_files()

        if self._dataframe is None:
            self.build_dataframe()

        return self._dataframe

    @property
    def label_columns(self):
        '''
        Keep column list for labels in metadata to persist through saving
        '''
        return self.config.get('label_columns', [])

    @property
    def X(self):
        '''
        Return the subset that isn't in the target labels
        '''
        return self.dataframe[self.dataframe.columns.difference(self.label_columns)]

    @property
    def y(self):
        '''
        Return the target label columns
        '''
        return self.dataframe[self.label_columns]

    def build_dataframe(self):
        '''
        Must set self._dataframe
        Cant set as abstractmethod because of database lookup dependency
        '''
        raise NotImplementedError

    def _hash(self):
        '''
        Datasets rely on external data so instead of hashing only the config,
        hash the actual resulting dataframe
        This requires loading the data before determining duplication
        so overwrite for differing behavior

        Hash is the combination of the:
            1) Dataframe
            2) Config
        '''
        return hash(self.custom_hasher((self.dataframe, self.config)))

    def _save_external_files(self):
        '''
        Shared method to save dataframe into a new table with name = GUID

        Hardcoded to only store in database so overwrite to use pickled
        objects or other storage mechanism
        '''
        self.filepaths = {"database": [(self._schema, str(self.id))]}
        self.df_to_sql(self._engine, self.dataframe,
                       str(self.id), schema=self._schema)

    def _load_external_files(self):
        '''
        Shared method to load dataframe from database

        Hardcoded to only pull from database so overwrite to use pickled
        objects or other storage mechanism
        '''
        schema, tablename = self.filepaths['database'][0]
        self._dataframe = self.load_sql(
            'select * from "{}"."{}"'.format(schema, tablename),
            self._engine
        )

        # Indicate externals were loaded
        self.unloaded_externals = False

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
