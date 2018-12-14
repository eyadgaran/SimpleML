'''
Setup testing env
'''

from simpleml.utils.initialization import Database
from subprocess import call
import random

__author__ = 'Elisha Yadgaran'

TEST_DATABASE = 'SimpleML-TEST-{}'.format(random.randint(10000, 99999))


def setup_package():
    print('Setting up testing env')
    Database(user='postgres', password='', database=TEST_DATABASE).initialize(drop_tables=True)


def teardown_package():
    print('Tearing Down')
    kill_connections = "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{}'".format(TEST_DATABASE)
    call(['psql', '-U', 'postgres', '-d', 'postgres', '-c', kill_connections])

    call(['psql', '-U', 'postgres', '-d', 'postgres', '-c',
          'DROP DATABASE "{}";'.format(TEST_DATABASE)])
