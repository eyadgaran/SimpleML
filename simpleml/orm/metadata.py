'''
Metadata bases specific to each session (subclasses represent tables affected by the
same session -- ie base.metadata.create_all()/drop_all()/upgrade())

Each class used as part of a session needs to be initialized directly
'''

__author__ = "Elisha Yadgaran"


import logging

from sqlalchemy import event, MetaData, DDL

from .base_sqlalchemy import BaseSQLAlchemy


LOGGER = logging.getLogger(__name__)


class SimplemlCoreSqlalchemy(BaseSQLAlchemy):
    """
    Shared metadata for all tables that live in the main schema
    """

    __abstract__ = True
    # Uses main (public) schema
    metadata = MetaData()


class BinaryStorageSqlalchemy(BaseSQLAlchemy):
    """
    Shared metadata for all tables that live in the binary storage schema
    """

    __abstract__ = True
    # Store binary data in its own schema
    SCHEMA = "BINARY"
    metadata = MetaData(schema=SCHEMA)

    @event.listens_for(metadata, "before_create", propagate=True)
    def _receive_before_create(target, connection, **kwargs):
        """
        Listen for and creates a new schema for datasets
        """
        # SQLite supports schemas as "Attached Databases"
        # https://www.sqlite.org/lang_attach.html
        if connection.dialect.name == "postgresql":
            LOGGER.debug("Issuing create schema if not exists")
            DDL("""CREATE SCHEMA IF NOT EXISTS "{}";""".format(target.schema)).execute(
                connection
            )

        # elif connection.dialect.name == 'sqlite':
        #     raise NotImplementedError('SQLite does not support multiple schemas right now')
        # TODO: Figure out a mechanism to pass in a dynamic schema so testing
        # doesnt mess with existing ones
        # DDL('''ATTACH DATABASE "{}";'''.format(target.schema))
        else:
            raise NotImplementedError(
                "Schemas not supported on {dialect}".format(
                    dialect=connection.dialect.name
                )
            )


class DatasetStorageSqlalchemy(BaseSQLAlchemy):
    """
    Shared metadata for all tables that live in the dataset storage schema
    """

    __abstract__ = True
    # Use different schemas/databases for storage optionality (dataframes are big in size)
    SCHEMA = "DATASETS"
    metadata = MetaData(schema=SCHEMA)

    @event.listens_for(metadata, "before_create", propagate=True)
    def _receive_before_create(target, connection, **kwargs):
        """
        Listen for and creates a new schema for datasets
        """
        # SQLite supports schemas as "Attached Databases"
        # https://www.sqlite.org/lang_attach.html
        if connection.dialect.name == "postgresql":
            LOGGER.debug("Issuing create schema if not exists")
            DDL("""CREATE SCHEMA IF NOT EXISTS "{}";""".format(target.schema)).execute(
                connection
            )

        # elif connection.dialect.name == 'sqlite':
        #     raise NotImplementedError('SQLite does not support multiple schemas right now')
        # TODO: Figure out a mechanism to pass in a dynamic schema so testing
        # doesnt mess with existing ones
        # DDL('''ATTACH DATABASE "{}";'''.format(target.schema))
        else:
            raise NotImplementedError(
                "Schemas not supported on {dialect}".format(
                    dialect=connection.dialect.name
                )
            )
