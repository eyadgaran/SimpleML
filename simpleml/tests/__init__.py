'''
Setup testing env

Assumes user `simpleml` exists with password `simpleml`
'''

__author__ = 'Elisha Yadgaran'


from simpleml.utils.initialization import Database
from simpleml.utils.postgres import create_database, drop_database
import random


TEST_DATABASE = 'SimpleML-TEST-{}'.format(random.randint(10000, 99999))
ADMIN_CONNECTION_PARAMS = {  # psycopg2.connect parameters
    'user': 'simpleml', 'password': 'simpleml', 'host': 'localhost',
    'port': 5432, 'database': 'postgres'
}
CONNECTION_PARAMS = {  # sqlalchemy.URL parameters
    'username': 'simpleml', 'password': 'simpleml', 'host': 'localhost',
    'port': 5432, 'database': TEST_DATABASE, 'drivername': 'postgresql'
}


def setup_package():
    print('Setting up testing env')
    create_database(ADMIN_CONNECTION_PARAMS, TEST_DATABASE)
    Database(**CONNECTION_PARAMS).initialize(create_tables=True, drop_tables=True)


def teardown_package():
    print('Tearing Down')
    drop_database(ADMIN_CONNECTION_PARAMS, TEST_DATABASE, force=True)
