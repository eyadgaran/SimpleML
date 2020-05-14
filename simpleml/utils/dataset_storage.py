'''
Optional module to persist dataset dataframes in database instead of filesystem
'''

__author__ = 'Elisha Yadgaran'


import logging

from simpleml.persistables.base_sqlalchemy import BaseSQLAlchemy
from sqlalchemy import MetaData, event, DDL


LOGGER = logging.getLogger(__name__)
# Use different schemas/databases for storage optionality (dataframes are big in size)
DATASET_SCHEMA = 'DATASETS'


class DatasetStorage(BaseSQLAlchemy):
    __abstract__ = True
    metadata = MetaData(schema=DATASET_SCHEMA)

    @event.listens_for(metadata, 'before_create', propagate=True)
    def _receive_before_create(target, connection, **kwargs):
        """
        Listen for and creates a new schema for datasets
        """
        # SQLite supports schemas as "Attached Databases"
        # https://www.sqlite.org/lang_attach.html
        if connection.dialect.name == 'postgresql':
            LOGGER.debug('Issuing create schema if not exists')
            DDL('''CREATE SCHEMA IF NOT EXISTS "{}";'''.format(DATASET_SCHEMA)).execute(connection)

        # elif connection.dialect.name == 'sqlite':
        #     raise NotImplementedError('SQLite does not support multiple schemas right now')
            # TODO: Figure out a mechanism to pass in a dynamic schema so testing
            # doesnt mess with existing ones
            # DDL('''ATTACH DATABASE "{}";'''.format(DATASET_SCHEMA))
        else:
            raise NotImplementedError('Schemas not supported on {dialect}'.format(dialect=connection.dialect.name))
