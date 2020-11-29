'''
Test suites that span the full implementation (end-to-end)

Integration tests assume a functional environment with active
database connections
'''

__author__ = 'Elisha Yadgaran'


import random
import unittest
import os
import glob

from simpleml.utils.configuration import FILESTORE_DIRECTORY
from simpleml.utils.initialization import Database
from simpleml.utils.postgres import create_database, drop_database
from simpleml.persistables.base_sqlalchemy import SimplemlCoreSqlalchemy, DatasetStorageSqlalchemy


DIRECTORY_IMPORT_PATH = 'simpleml.tests.integration'
DIRECTORY_ABSOLUTE_PATH = os.path.dirname(__file__)


class IntegrationTestSuite(unittest.TestSuite):
    def __init__(self):
        # Load all tests in the directory
        loader = unittest.TestLoader()
        module_paths = glob.glob(os.path.join(DIRECTORY_ABSOLUTE_PATH, 'test*.py'), recursive=True)  # /root/abspath/tests/test*.py
        relative_module_paths = [i[len(DIRECTORY_ABSOLUTE_PATH) + 1:] for i in module_paths]  # find the delta from this filepath test*.py
        module_imports = ['.'.join((DIRECTORY_IMPORT_PATH, i.replace('/', '.')[:-3])) for i in relative_module_paths]  # Import string simpleml.tests.test*
        tests = [loader.loadTestsFromName(i) for i in module_imports]
        super().__init__(tests=tests)

    def setUp(self):
        print('Running Integration Tests')

    def tearDown(self):
        print('Finished Running Integration Tests')

    def run(self, *args, **kwargs):
        self.setUp()
        super().run(*args, **kwargs)
        self.tearDown()


class SqliteIntegrationTestSuite(IntegrationTestSuite):
    '''
    Integration tests with sqlite fixture
    '''

    def __init__(self):
        # Specify the fixture configuration
        database_name = 'SimpleML-TEST-{}'.format(random.randint(10000, 99999))
        self.database_path = os.path.join(FILESTORE_DIRECTORY, '{}.db'.format(database_name))
        self.connection_params = {  # sqlalchemy.URL parameters
            'uri': 'sqlite:///{}'.format(self.database_path)
            # TODO: In-memory sqlite database - doesnt require teardown
        }
        super().__init__()

    def setUp(self):
        print('Running SQLite Integration Tests')
        print(self.database_path)
        Database(**self.connection_params).initialize(
            base_list=[SimplemlCoreSqlalchemy],
            upgrade=True, create_tables=True, drop_tables=True)

    def tearDown(self):
        print('Finished Running SQLite Integration Tests')
        os.remove(self.database_path)


class PostgresIntegrationTestSuite(IntegrationTestSuite):
    '''
    Integration tests with Postgres fixture
    '''

    def __init__(self):
        # Specify the fixture configuration
        self.database_name = 'SimpleML-TEST-{}'.format(random.randint(10000, 99999))
        # Postgres requires an admin role for elevated permissions to create/drop databases
        self.admin_connection_params = {  # psycopg2.connect parameters
            'user': 'simpleml', 'password': 'simpleml', 'host': 'localhost',
            'port': 5432, 'database': 'postgres'
        }
        self.connection_params = {  # sqlalchemy.URL parameters
            'username': 'simpleml', 'password': 'simpleml', 'host': 'localhost',
            'port': 5432, 'database': self.database_name, 'drivername': 'postgresql'
        }
        super().__init__()

    def setUp(self):
        print('Running Postgres Integration Tests')
        print(self.database_name)
        create_database(self.admin_connection_params, self.database_name)
        Database(**self.connection_params).initialize(
            base_list=[SimplemlCoreSqlalchemy, DatasetStorageSqlalchemy],
            upgrade=True, create_tables=True, drop_tables=True)

    def tearDown(self):
        print('Finished Running Postgres Integration Tests')
        drop_database(self.admin_connection_params, self.database_name, force=True)


def load_tests(*args, **kwargs):
    integration_tests = unittest.TestSuite()
    integration_tests.addTests((SqliteIntegrationTestSuite(), PostgresIntegrationTestSuite()))
    return integration_tests


def run_tests():
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(load_tests())


if __name__ == '__main__':
    run_tests()
