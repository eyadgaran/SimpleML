'''
Util module to manage postgres specific functions
'''

__author__ = 'Elisha Yadgaran'


from simpleml import psycopg2


def create_database(connection_params, database, owner=None):
    '''
    Creates a new database
    :return: None
    '''
    owner_syntax = ' WITH OWNER {}'.format(owner) if owner else ''
    database_command = 'CREATE DATABASE "{database}" {owner};'.format(
        database=database, owner=owner_syntax)
    try:
        run_sql_command(connection_params, database_command, autocommit=True)
    except psycopg2.ProgrammingError:
        pass


def create_user(connection_params, user, password):
    '''
    Creates a new user
    :return: None
    '''
    user_command = "CREATE USER {user} PASSWORD '{password}';".format(
        user=user, password=password)
    try:
        run_sql_command(connection_params, user_command, autocommit=True)
    except psycopg2.ProgrammingError:
        pass


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