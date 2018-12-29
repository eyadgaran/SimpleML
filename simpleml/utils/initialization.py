'''
Util module to initialize SimpleML and configure
database management
'''

__author__ = 'Elisha Yadgaran'


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
from simpleml.persistables.serializing import custom_dumps, custom_loads

from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import scoped_session, sessionmaker
import logging

LOGGER = logging.getLogger(__name__)


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
        return create_engine(self.engine_url,
                             json_serializer=custom_dumps,
                             json_deserializer=custom_loads,
                             pool_recycle=300)

    @staticmethod
    def create_tables(base, drop_tables=False):
        '''
        Creates database tables.

        :param drop_tables: Whether or not to drop the existing tables first.
        :return: None
        '''
        if drop_tables:
            base.metadata.drop_all()

        try:
            base.metadata.create_all()
        except ProgrammingError as e:  # Permission errors
            LOGGER.debug(e)

    def _initialize(self, base, drop_tables):
        '''
        Initialization method to set up database connection and inject
        session manager

        :param drop_tables: Bool, whether to drop existing tables in database
        :return: None
        '''
        engine = self.engine
        session = scoped_session(sessionmaker(autocommit=True,
                                              autoflush=False,
                                              bind=engine))
        base.metadata.bind = engine
        base.query = session.query_property()

        self.create_tables(base, drop_tables=drop_tables)

        base.set_session(session)

    def initialize(self, base_list=None, drop_tables=False):
        '''
        Initialization method to set up database connection and inject
        session manager

        :param drop_tables: Bool, whether to drop existing tables in database
        :return: None
        '''
        if base_list is None:
            base_list = [BasePersistable, DatasetStorage, RawDatasetStorage, BinaryBlob]

        for base in base_list:
            self._initialize(base, drop_tables=drop_tables)
