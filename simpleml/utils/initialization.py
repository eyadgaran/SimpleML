'''
Util module to initialize SimpleML and configure
database management
'''

# Import table models to register in DeclaritiveBase
from simpleml.persistables.base_persistable import BasePersistable
import simpleml.datasets.raw_datasets.base_raw_dataset
import simpleml.pipelines.dataset_pipelines.base_dataset_pipeline
import simpleml.datasets.processed_datasets.base_processed_dataset
import simpleml.pipelines.production_pipelines.base_production_pipeline
import simpleml.models.base_model
import simpleml.metrics.base_metric
from simpleml.persistables.dataset_storage import DatasetStorage, RawDatasetStorage
from simpleml.persistables.binary_blob import BinaryBlob

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
import psycopg2
from psycopg2 import ProgrammingError


__author__ = 'Elisha Yadgaran'


class Database(object):
    '''
    Basic configuration to interact with database
    '''
    def __init__(self, database='SimpleML', user='simpleml',
                 password='simpleml', jdbc='postgresql',
                 host='localhost', port=5432):
        self.database_params = {
            'database': database,
            'user': user,
            'password': password,
            'jdbc': jdbc,
            'host': host,
            'port': port
        }

    @property
    def database_name(self):
        return self.database_params.get('database')

    @property
    def database_user(self):
        return self.database_params.get('user')

    @property
    def database_password(self):
        return self.database_params.get('password')

    @property
    def engine_url(self):
        return '{jdbc}://{user}:{password}@{host}:{port}/{database}'.format(
            **self.database_params)

    @property
    def engine(self):
        return create_engine(self.engine_url)

    @staticmethod
    def create_tables(base, drop_tables=False):
        '''
        Creates database tables.

        :param drop_tables: Whether or not to drop the existing tables first.
        :return: None
        '''
        if drop_tables:
            base.metadata.drop_all()

        base.metadata.create_all()

    def create_database(self):
        '''
        Creates database via command line.
        TODO: make system and sql flavor agnostic

        :return: None
        '''
        admin_params = {'user': 'postgres', 'database': 'postgres',
                        'host': self.database_params.get('host'),
                        'port': self.database_params.get('port')}
        user_command = "CREATE USER {user} PASSWORD '{password}';".format(
            user=self.database_user, password=self.database_password)
        database_command =  'CREATE DATABASE "{database}" WITH OWNER {user};'.format(
             database=self.database_name, user=self.database_user)

        try:
            run_sql_command(admin_params, user_command, autocommit=True)
        except ProgrammingError:
            pass
        try:
            run_sql_command(admin_params, database_command, autocommit=True)
        except ProgrammingError:
            pass

    def _initialize(self, base, create_database, drop_tables):
        '''
        Initialization method to set up database connection and inject
        session manager

        :param create_database: Bool, whether to run database and user creation
            calls before starting up
        :param drop_tables: Bool, whether to drop existing tables in database
        :return: None
        '''
        engine = self.engine
        session = scoped_session(sessionmaker(autocommit=True,
                                              autoflush=False,
                                              bind=engine))
        base.metadata.bind = engine
        base.query = session.query_property()

        if create_database:
            self.create_database()

        self.create_tables(base, drop_tables=drop_tables)

        base.set_session(session)

    def initialize(self, base_list=None, create_database=True, drop_tables=False):
        '''
        Initialization method to set up database connection and inject
        session manager

        :param create_database: Bool, whether to run database and user creation
            calls before starting up
        :param drop_tables: Bool, whether to drop existing tables in database
        :return: None
        '''
        if base_list is None:
            base_list = [BasePersistable, DatasetStorage, RawDatasetStorage, BinaryBlob]

        for base in base_list:
            self._initialize(base, create_database=create_database, drop_tables=drop_tables)
            # Only create on the first go
            create_database = False

def run_sql_command(connection_params, command, autocommit=False):
    '''
    Execute command directly using psycopg2 cursor

    :param connection_params: dict of connection details
    :param command: raw sql to execute
    :param autocommit: default false; determines if the connection automcommits
    commands. Necessary for certain commands (create/drop db)
    '''
    connection = psycopg2.connect(**connection_params)
    cursor = connection.cursor()
    connection.autocommit = autocommit

    cursor.execute(command)

    if not autocommit:
        connection.commit()

    cursor.close()
    connection.close()
