'''
Base class for sqlalchemy
'''

__author__ = 'Elisha Yadgaran'

import logging

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, DateTime, func, event, MetaData, DDL
from sqlalchemy_mixins import AllFeaturesMixin


Base = declarative_base()
LOGGER = logging.getLogger(__name__)


class BaseSQLAlchemy(Base, AllFeaturesMixin):
    '''
    Base class for all SimpleML database objects. Defaults to PostgreSQL
    but can be swapped out for any supported SQLAlchemy backend.

    Takes advantage of sqlalchemy-mixins to enable active record operations
    (TableModel.save(), create(), where(), destroy())

    Added some inheritable convenience methods

    -------
    Schema
    -------
    created_timestamp: Server time on insert
    modified_timestamp: Server time on update
    '''
    __abstract__ = True

    created_timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    modified_timestamp = Column(DateTime(timezone=True), server_onupdate=func.now())

    @classmethod
    def filter(cls, *filters):
        return cls._session.query(cls).filter(*filters)

    @classmethod
    def query_by(cls, *queries):
        return cls._session.query(*queries)


@event.listens_for(BaseSQLAlchemy, 'before_update', propagate=True)
def _receive_before_update(mapper, connection, target):
    """Listen for updates and update `modified_timestamp` column."""
    target.modified_timestamp = func.now()


'''
Metadata bases specific to each session (subclasses represent tables affected by the
same session -- ie base.metadata.create_all()/drop_all()/upgrade())
'''


class SimplemlCoreSqlalchemy(BaseSQLAlchemy):
    '''
    Shared metadata for all tables that live in the main schema
    '''
    __abstract__ = True
    # Uses main (public) schema
    metadata = MetaData()


class BinaryStorageSqlalchemy(BaseSQLAlchemy):
    '''
    Shared metadata for all tables that live in the binary storage schema
    '''
    __abstract__ = True
    # Store binary data in its own schema
    SCHEMA = 'BINARY'
    metadata = MetaData(schema=SCHEMA)

    @event.listens_for(metadata, 'before_create', propagate=True)
    def _receive_before_create(target, connection, **kwargs):
        """
        Listen for and creates a new schema for datasets
        """
        # SQLite supports schemas as "Attached Databases"
        # https://www.sqlite.org/lang_attach.html
        if connection.dialect.name == 'postgresql':
            LOGGER.debug('Issuing create schema if not exists')
            DDL('''CREATE SCHEMA IF NOT EXISTS "{}";'''.format(target.schema)).execute(connection)

        # elif connection.dialect.name == 'sqlite':
        #     raise NotImplementedError('SQLite does not support multiple schemas right now')
            # TODO: Figure out a mechanism to pass in a dynamic schema so testing
            # doesnt mess with existing ones
            # DDL('''ATTACH DATABASE "{}";'''.format(target.schema))
        else:
            raise NotImplementedError('Schemas not supported on {dialect}'.format(dialect=connection.dialect.name))


class DatasetStorageSqlalchemy(BaseSQLAlchemy):
    '''
    Shared metadata for all tables that live in the dataset storage schema
    '''
    __abstract__ = True
    # Use different schemas/databases for storage optionality (dataframes are big in size)
    SCHEMA = 'DATASETS'
    metadata = MetaData(schema=SCHEMA)

    @event.listens_for(metadata, 'before_create', propagate=True)
    def _receive_before_create(target, connection, **kwargs):
        """
        Listen for and creates a new schema for datasets
        """
        # SQLite supports schemas as "Attached Databases"
        # https://www.sqlite.org/lang_attach.html
        if connection.dialect.name == 'postgresql':
            LOGGER.debug('Issuing create schema if not exists')
            DDL('''CREATE SCHEMA IF NOT EXISTS "{}";'''.format(target.schema)).execute(connection)

        # elif connection.dialect.name == 'sqlite':
        #     raise NotImplementedError('SQLite does not support multiple schemas right now')
            # TODO: Figure out a mechanism to pass in a dynamic schema so testing
            # doesnt mess with existing ones
            # DDL('''ATTACH DATABASE "{}";'''.format(target.schema))
        else:
            raise NotImplementedError('Schemas not supported on {dialect}'.format(dialect=connection.dialect.name))
