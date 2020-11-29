'''
Module for save patterns registered for database persistence
'''

__author__ = 'Elisha Yadgaran'


import pandas as pd
from typing import Any, Dict

from simpleml.persistables.base_sqlalchemy import DatasetStorageSqlalchemy
from simpleml.save_patterns.decorators import SavePatternDecorators
from simpleml.save_patterns.base import BaseSavePattern
from simpleml.utils.binary_blob import BinaryBlob


@SavePatternDecorators.register_save_pattern
class DatabaseTableSavePattern(BaseSavePattern):
    '''
    Save pattern implementation to save dataframes to a database table
    '''
    SAVE_PATTERN = 'database_table'

    @classmethod
    def save(cls,
             obj: pd.DataFrame,
             persistable_id: str,
             schema: str = DatasetStorageSqlalchemy.SCHEMA,
             **kwargs) -> Dict[str, str]:
        '''
        Save method to save dataframe into a new table with name = GUID
        Updates filepath for the artifact with the schema and table
        '''
        engine = DatasetStorageSqlalchemy.metadata.bind
        cls.df_to_sql(engine, df=obj, table=persistable_id, schema=schema)

        return {'schema': schema, 'table': persistable_id}

    @classmethod
    def load(cls,
             filepath_data: Dict[str, str],
             **kwargs) -> pd.DataFrame:
        '''
        Load method to load dataframe from database
        '''
        schema = filepath_data['schema']
        table = filepath_data['table']
        engine = DatasetStorageSqlalchemy.metadata.bind
        df = cls.load_sql(
            'select * from "{}"."{}"'.format(schema, table),
            engine
        )

        return df


@SavePatternDecorators.register_save_pattern
class DatabasePickleSavePattern(BaseSavePattern):
    '''
    Save pattern implementation to save binary objects to a database table
    '''
    SAVE_PATTERN = 'database_pickled'

    @classmethod
    def save(cls,
             obj: Any,
             persistable_type: str,
             persistable_id: str,
             **kwargs) -> str:
        '''
        Save method to save files into binary schema

        Hardcoded to only store pickled objects in database so overwrite to use
        other storage mechanism
        '''
        pickled_stream = cls.pickle_object(obj, as_stream=True)
        pickled_record = BinaryBlob.create(
            object_type=persistable_type, object_id=persistable_id, binary_blob=pickled_stream)
        return str(pickled_record.id)

    @classmethod
    def load(cls,
             primary_key: str,
             **kwargs) -> Any:
        '''
        Load method to load files from database

        Hardcoded to only pull from pickled so overwrite to use
        other storage mechanism
        '''
        pickled_stream = BinaryBlob.find(primary_key).binary_blob
        return cls.load_pickled_object(pickled_stream, stream=True)
