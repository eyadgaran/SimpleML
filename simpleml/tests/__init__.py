'''
Setup testing env
'''

__author__ = 'Elisha Yadgaran'


from simpleml.utils.initialization import Database
from simpleml.utils.postgres import create_database
from subprocess import call
import random


TEST_DATABASE = 'SimpleML-TEST-{}'.format(random.randint(10000, 99999))


def setup_package():
    print('Setting up testing env')
    connection_params = {
        'user': 'postgres', 'password': '', 'host': 'localhost',
        'port': 5432, 'database': 'postgres'
    }
    create_database(connection_params, TEST_DATABASE)
    Database(user='postgres', password='', database=TEST_DATABASE).initialize(drop_tables=True)


def teardown_package():
    print('Tearing Down')
    kill_connections = "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{}'".format(TEST_DATABASE)
    call(['psql', '-U', 'postgres', '-d', 'postgres', '-c', kill_connections])

    call(['psql', '-U', 'postgres', '-d', 'postgres', '-c',
          'DROP DATABASE "{}";'.format(TEST_DATABASE)])
