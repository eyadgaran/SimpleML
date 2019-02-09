from __future__ import with_statement

from logging.config import fileConfig

from alembic import context
import os

from simpleml.utils.initialization import Database
from simpleml.utils.errors import SimpleMLError
from simpleml.persistables.base_persistable import Persistable

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
if not Persistable.metadata.is_bound():
    # Initialize a new session if one isn't already configured
    connection_params = {
        'username': os.getenv('SIMPLEML_DATABASE_USERNAME'),
        'password': os.getenv('SIMPLEML_DATABASE_PASSWORD'),
        'database': os.getenv('SIMPLEML_DATABASE_NAME'),
        'drivername': os.getenv('SIMPLEML_DATABASE_DRIVERNAME'),
        'host': os.getenv('SIMPLEML_DATABASE_HOST'),
        'port': os.getenv('SIMPLEML_DATABASE_PORT'),
    }
    missing_params = ['SIMPLEML_DATABASE_' + (k.upper() if k != 'database' else 'NAME')
                      for k, v in connection_params.items() if v is None]
    if missing_params:
        raise SimpleMLError('Missing Connection Parameters (env variables): {}'.format(', '.join(missing_params)))
    Database(**connection_params).initialize(base_list=[Persistable])

target_metadata = Persistable.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline():
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = 'postgresql://user:pass@localhost/dbname'
    context.configure(
        url=url, target_metadata=target_metadata, literal_binds=True,
        compare_type=True
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = target_metadata.bind

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata,
            compare_type=True
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
