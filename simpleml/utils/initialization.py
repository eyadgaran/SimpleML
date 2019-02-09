'''
Util module to initialize SimpleML and configure
database management
'''

__author__ = 'Elisha Yadgaran'


# Import table models to register in DeclaritiveBase
from simpleml.persistables.base_persistable import Persistable
import simpleml.datasets.base_dataset
import simpleml.pipelines.base_pipeline
import simpleml.models.base_model
import simpleml.metrics.base_metric
from simpleml.persistables.dataset_storage import DatasetStorage
from simpleml.persistables.binary_blob import BinaryBlob
from simpleml.persistables.serializing import custom_dumps, custom_loads

from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.engine.url import URL
from alembic.config import Config
from alembic import command
from os.path import realpath, dirname, join
import logging

LOGGER = logging.getLogger(__name__)


class Database(URL):
    '''
    Basic configuration to interact with database
    '''
    def __init__(self, database='SimpleML', username='simpleml',
                 password='simpleml', drivername='postgresql',
                 host='localhost', port=5432, **kwargs):
        super(Database, self).__init__(
            drivername=drivername,
            username=username,
            password=password,
            host=host,
            port=port,
            database=database,
            **kwargs
        )

    @property
    def engine(self):
        return create_engine(self,
                             json_serializer=custom_dumps,
                             json_deserializer=custom_loads,
                             pool_recycle=300)

    @property
    def alembic_config(self):
        if not hasattr(self, '_alembic_config'):
            # load the Alembic configuration
            root_path = dirname(dirname(dirname(realpath(__file__))))
            self._alembic_config = Config(join(root_path, 'alembic.ini'))
            # For some reason, alembic doesnt use a relative path from the ini
            # and cannot find the migration folder without the full path
            self._alembic_config.set_main_option('script_location', join(root_path, 'migrations'))
        return self._alembic_config

    def create_tables(self, base, drop_tables=False, ignore_errors=False):
        '''
        Creates database tables (and potentially drops existing ones).
        Assumes to be running under a sufficiently privileged user

        :param drop_tables: Whether or not to drop the existing tables first.
        :return: None
        '''
        try:
            if drop_tables:
                base.metadata.drop_all()
                # downgrade the version table, "stamping" it with the base rev
                command.stamp(self.alembic_config, "base")

            base.metadata.create_all()
            # generate/upgrade the version table, "stamping" it with the most recent rev
            command.stamp(self.alembic_config, "head")

        except ProgrammingError as e:  # Permission errors
            if ignore_errors:
                LOGGER.debug(e)
            else:
                raise(e)

    def upgrade(self, revision='head'):
        '''
        Proxy Method to invoke alembic upgrade command to specified revision
        '''
        command.upgrade(self.alembic_config, revision)

    def downgrade(self, revision):
        '''
        Proxy Method to invoke alembic downgrade command to specified revision
        '''
        command.downgrade(self.alembic_config, revision)

    def _initialize(self, base, create_tables=False, **kwargs):
        '''
        Initialization method to set up database connection and inject
        session manager

        :param create_tables: Bool, whether to create tables in database
        :param drop_tables: Bool, whether to drop existing tables in database
        :return: None
        '''
        engine = self.engine
        session = scoped_session(sessionmaker(autocommit=True,
                                              autoflush=False,
                                              bind=engine))
        base.metadata.bind = engine
        base.query = session.query_property()

        if create_tables:
            self.create_tables(base, **kwargs)

        base.set_session(session)

    def initialize(self, base_list=None, **kwargs):
        '''
        Initialization method to set up database connection and inject
        session manager

        :param drop_tables: Bool, whether to drop existing tables in database
        :return: None
        '''
        if base_list is None:  # Use defaults in project
            base_list = [Persistable]

        for base in base_list:
            self._initialize(base, **kwargs)
