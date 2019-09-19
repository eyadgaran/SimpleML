'''
Util module to manage postgres specific functions
'''

__author__ = 'Elisha Yadgaran'


from simpleml.imports import psycopg2


def create_database(connection_params, database, owner=None, raise_error=True):
    '''
    Creates a new database
    :return: None
    '''
    owner_syntax = ' WITH OWNER {}'.format(owner) if owner else ''
    database_command = 'CREATE DATABASE "{database}" {owner};'.format(
        database=database, owner=owner_syntax)
    try:
        run_sql_command(connection_params, database_command, autocommit=True)
    except psycopg2.ProgrammingError as e:
        if raise_error:
            raise(e)


def create_user(connection_params, user, password, raise_error=True):
    '''
    Creates a new user
    :return: None
    '''
    user_command = "CREATE USER {user} PASSWORD '{password}';".format(
        user=user, password=password)
    try:
        run_sql_command(connection_params, user_command, autocommit=True)
    except psycopg2.ProgrammingError as e:
        if raise_error:
            raise(e)


def drop_database(connection_params, database, force=False, raise_error=True):
    '''
    Drop database -- Must have sufficient privileges
    :return: None
    '''
    force_command = "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{}'".format(database)
    database_command = 'DROP DATABASE "{database}";'.format(database=database)

    try:
        if force:
            run_sql_command(connection_params, force_command)
        run_sql_command(connection_params, database_command, autocommit=True)
    except psycopg2.ProgrammingError as e:
        if raise_error:
            raise(e)


def drop_user(connection_params, user, raise_error=True):
    '''
    Drop a user -- Must have sufficient privileges
    :return: None
    '''
    user_command = "DROP USER {user};".format(user=user)
    try:
        run_sql_command(connection_params, user_command, autocommit=True)
    except psycopg2.ProgrammingError as e:
        if raise_error:
            raise(e)


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
