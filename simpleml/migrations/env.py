'''
Central place to actually run any migrations. Can be invoked in a few ways:
    1) simpleml upgrade/downgrade/etc
    2) alembic upgrade/downgrade/etc WITH env.ALEMBIC_CONFIG pointing to here OR called from this directory
    3) in a python session via `db.upgrade/downgrade`
'''

from __future__ import with_statement
from logging.config import fileConfig
from alembic import context

from simpleml.utils.initialization import Database
from simpleml.persistables.base_sqlalchemy import SimplemlCoreSqlalchemy

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
fileConfig(config.config_file_name, disable_existing_loggers=False)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
if not SimplemlCoreSqlalchemy.metadata.is_bound():
    # Initialize a new session if one isn't already configured
    # Do not validate schema since it will be out of sync
    Database().initialize(base_list=[SimplemlCoreSqlalchemy], validate=False)

target_metadata = SimplemlCoreSqlalchemy.metadata

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
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # Allow config object to have a connection already added
    connectable = config.attributes.get('connection', None)

    if connectable is None:
        # only create Engine if we don't have a Connection
        # from the outside
        connectable = target_metadata.bind

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            transaction_per_migration=True,
            render_as_batch=True  # for SQLite support: https://alembic.sqlalchemy.org/en/latest/batch.html
        )

        with context.begin_transaction():
            context.run_migrations()


if __name__ == 'env_py':
    # alembic entrypoint is env_py
    # regular import would be env
    if context.is_offline_mode():
        run_migrations_offline()
    else:
        run_migrations_online()
