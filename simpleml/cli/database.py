'''
CLI functions for database management
'''

__author__ = 'Elisha Yadgaran'


import logging

import click

from simpleml.persistables.base_sqlalchemy import SimplemlCoreSqlalchemy
from simpleml.utils.initialization import Database

LOGGER = logging.getLogger(__name__)


@click.group()
def db():
    '''
    Entrypoint for database related operations
    '''


def _init_connection() -> Database:
    '''
    Binds the db engine and initializes a connection
    '''
    LOGGER.info('Initializing a SimpleML database connection. CLI only supports env variable database parameters (`SIMPLEML_DATABASE_*`)')
    db = Database()

    if not SimplemlCoreSqlalchemy.metadata.is_bound():
        # Initialize a new session if one isn't already configured
        # Do not validate schema since it will be out of sync
        db.initialize(base_list=[SimplemlCoreSqlalchemy], validate=False)
    return db


@db.command('init', short_help='Initializes a new database with the latest tables and schemas')
def initialize():
    '''
    Initializes a new database with the latest tables and schemas
    (equivalent to `db.initialize(create_tables=True, upgrade=True)`)
    '''
    db = _init_connection()
    db.create_tables(base=SimplemlCoreSqlalchemy)


@db.command('upgrade', short_help='Upgrades database to revision')
@click.option('--revision', '-r', default='head', show_default=True, help='The alembic revision to upgrade to')
def upgrade(revision):
    '''
    Upgrades database to revision
    '''
    db = _init_connection()
    db.upgrade(revision=revision)


@db.command('downgrade', short_help='Downgrades database to revision')
@click.argument('revision')
def downgrade(revision):
    '''
    Downgrades database to revision

    REVISION should be the alembic revision to downgrade to
    '''
    db = _init_connection()
    db.downgrade(revision=revision)


def nuke():
    '''
    Full wipe of the database
    (equivalent to `base.metadata.drop_all()`)
    '''
