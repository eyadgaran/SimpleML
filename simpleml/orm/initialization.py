"""
Util module to initialize SimpleML and configure
database management
"""

__author__ = "Elisha Yadgaran"

import atexit
import logging
import os
import random
from os.path import dirname, join, realpath
from typing import Any, Dict, Optional, Tuple

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine
from sqlalchemy.engine.url import URL, make_url
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import scoped_session, sessionmaker

from simpleml.imports import SSHTunnelForwarder

# Import table models to register in DeclaritiveBase
from simpleml.orm.metadata import (
    BinaryStorageSqlalchemy,
    DatasetStorageSqlalchemy,
    SimplemlCoreSqlalchemy,
)
from simpleml.orm.serializing import custom_dumps, custom_loads
from simpleml.utils.configuration import CONFIG, FILESTORE_DIRECTORY
from simpleml.utils.errors import SimpleMLError

LOGGER = logging.getLogger(__name__)


# Database Defaults
DATABASE_NAME = os.getenv("SIMPLEML_DATABASE_NAME", None)
DATABASE_USERNAME = os.getenv("SIMPLEML_DATABASE_USERNAME", None)
DATABASE_PASSWORD = os.getenv("SIMPLEML_DATABASE_PASSWORD", None)
DATABASE_HOST = os.getenv("SIMPLEML_DATABASE_HOST", None)
DATABASE_PORT = os.getenv("SIMPLEML_DATABASE_PORT", None)
DATABASE_DRIVERNAME = os.getenv("SIMPLEML_DATABASE_DRIVERNAME", None)
DATABASE_QUERY = os.getenv("SIMPLEML_DATABASE_QUERY", None)
DATABASE_CONF = os.getenv("SIMPLEML_DATABASE_CONF", None)
DATABASE_URI = os.getenv("SIMPLEML_DATABASE_URI", None)


class BaseDatabase(object):
    """
    Base Database class to configure db connection
    Does not assume schema tracking or any other validation

    Starting in sqlalchemy 1.4.2, the signature of `sqlalchemy.engine.url.URL`
    has changed to an immutable object without an __init__
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        configuration_section: Optional[str] = None,
        uri: Optional[str] = None,
        use_ssh_tunnel: bool = False,
        sshtunnel_params: Optional[Dict[str, Any]] = None,
        **credentials,
    ):
        """
        :param use_ssh_tunnel: boolean - default false. Whether to tunnel sqlalchemy connection
            through an ssh tunnel or not
        :param sshtunnel_params: Dict of ssh params - specify according to sshtunnel project
            https://github.com/pahaz/sshtunnel/ - direct passthrough
        """
        self.use_ssh_tunnel = use_ssh_tunnel

        # Sort out which credentials are the final ones -- default to remaining passed params
        if configuration_section is not None:
            if config is None:
                raise SimpleMLError("Cannot use config section without a config file")
            # Default to credentials in config file
            credentials = dict(config[configuration_section])
        elif uri is not None:
            # Deconstruct URI into credentials
            url = make_url(uri)
            credentials = {
                "drivername": url.drivername,
                "username": url.username,
                "password": url.password,
                "host": url.host,
                "port": url.port,
                "database": url.database,
                "query": url.query,
            }

        # Reconfigure credentials if SSH tunnel specified
        if self.use_ssh_tunnel:
            LOGGER.warning(
                """
                Usage: call Database.open_tunnel() before Database.initialize() and
                end script with Database.close_tunnel()
                Configure connection with supported parameters passed via
                `sshtunnel_params={**configs}`. Binding and routing through local
                port is automatically handled, but other parameters like `set_keepalive`
                may be interesting. https://sshtunnel.readthedocs.io/en/latest/
                """
            )
            # Overwrite passed ports and hosts to route localhost port to the
            # original destination via tunnel
            if sshtunnel_params is None:
                sshtunnel_params = {}
            credentials, self.ssh_config = self.configure_ssh_tunnel(
                credentials, sshtunnel_params
            )

        try:
            # New syntax (1.4.2+)
            self.url = URL.create(**credentials)
        except AttributeError:
            # Old syntax
            self.url = URL(**credentials)

    def configure_ssh_tunnel(
        self, credentials: Dict[str, Any], ssh_config: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Actual DB location
        target_host = credentials.pop("host")
        target_port = int(credentials.pop("port"))

        # SSH Tunnel location
        local_host, local_port = ssh_config.get("local_bind_address", (None, None))
        local_host = local_host or "localhost"  # In case it's null
        local_port = local_port or random.randint(4000, 5000)  # In case it's null
        LOGGER.info("Using {}:{} to bind SSH tunnel".format(local_host, local_port))

        # Swap em - db URI points to the local tunnel opening and the remote
        # ssh tunnel binds to the original host+port
        credentials["host"] = local_host
        credentials["port"] = local_port

        ssh_config["local_bind_address"] = (local_host, local_port)
        ssh_config["remote_bind_address"] = (target_host, target_port)

        return credentials, ssh_config

    def open_tunnel(self) -> None:
        atexit.register(self.close_tunnel)
        self.ssh_tunnel.start()

    def close_tunnel(self) -> None:
        self.ssh_tunnel.stop()

    @property
    def engine(self) -> Any:
        # Custom serializer/deserializer not supported by all drivers
        # Definitely works for:
        # - Postgres
        # - SQLite >= 1.3.7 -- Use _json_serializer for below
        return create_engine(
            self.url,
            json_serializer=custom_dumps,
            json_deserializer=custom_loads,
            pool_recycle=300,
        )

    @property
    def ssh_tunnel(self) -> SSHTunnelForwarder:
        if SSHTunnelForwarder is None:  # Not installed
            raise SimpleMLError(
                "SSHTunnel is not installed, install with `pip install sshtunnel`"
            )

        if not hasattr(self, "_sshtunnel"):
            self._sshtunnel = SSHTunnelForwarder(**self.ssh_config)
        return self._sshtunnel

    def create_tables(
        self, base, drop_tables: bool = False, ignore_errors: bool = False
    ) -> None:
        """
        Creates database tables (and potentially drops existing ones).
        Assumes to be running under a sufficiently privileged user

        :param drop_tables: Whether or not to drop the existing tables first.
        :return: None
        """
        try:
            if drop_tables:
                base.metadata.drop_all()

            base.metadata.create_all()

        except ProgrammingError as e:  # Permission errors
            if ignore_errors:
                LOGGER.debug(e)
            else:
                raise (e)

    def _initialize(self, base, create_tables=False, **kwargs):
        """
        Initialization method to set up database connection and inject
        session manager

        :param create_tables: Bool, whether to create tables in database
        :param drop_tables: Bool, whether to drop existing tables in database
        :return: None
        """
        engine = self.engine
        session = scoped_session(
            sessionmaker(autocommit=True, autoflush=False, bind=engine)
        )
        base.metadata.bind = engine
        base.query = session.query_property()

        if create_tables:
            self.create_tables(base, **kwargs)

        base.set_session(session)

    def initialize(self, base_list, **kwargs):
        """
        Initialization method to set up database connection and inject
        session manager

        Raises a SimpleML error if database schema is not up to date

        :param drop_tables: Bool, whether to drop existing tables in database
        :param upgrade: Bool, whether to run an upgrade migration after establishing a connection
        :return: None
        """
        for base in base_list:
            self._initialize(base, **kwargs)

    # delegate representation to URL
    def __str__(self):
        return self.url.__str__()

    def __repr__(self):
        return self.url.__repr__()


class AlembicDatabase(BaseDatabase):
    """
    Base database class to manage dbs with schema tracking. Includes alembic
    config references
    """

    def __init__(self, alembic_filepath, script_location="migrations", *args, **kwargs):
        self.alembic_filepath = alembic_filepath
        self.script_location = script_location
        super(AlembicDatabase, self).__init__(*args, **kwargs)

    @property
    def alembic_config(self):
        if not hasattr(self, "_alembic_config"):
            # load the Alembic configuration
            self._alembic_config = Config(self.alembic_filepath)
            # For some reason, alembic doesnt use a relative path from the ini
            # and cannot find the migration folder without the full path
            self._alembic_config.set_main_option(
                "script_location",
                join(dirname(self.alembic_filepath), self.script_location),
            )
        return self._alembic_config

    def create_tables(self, base, drop_tables=False, ignore_errors=False):
        """
        Creates database tables (and potentially drops existing ones).
        Assumes to be running under a sufficiently privileged user

        :param drop_tables: Whether or not to drop the existing tables first.
        :return: None
        """
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
                raise (e)

    def upgrade(self, revision="head"):
        """
        Proxy Method to invoke alembic upgrade command to specified revision
        Indirectly runs the alembic env.py code
        """
        command.upgrade(self.alembic_config, revision)

    def downgrade(self, revision):
        """
        Proxy Method to invoke alembic downgrade command to specified revision
        Indirectly runs the alembic env.py code
        """
        command.downgrade(self.alembic_config, revision)

    def validate_schema_version(self, base_list):
        """
        Check that the newly initialized database is up-to-date
        Raises an error otherwise (ahead of any table model mismatches later)
        """
        # Iterate base list and check schema against alembic config (if different
        # bases use different configs, they need to be invoked in different classes)
        # Does not actually trigger any migrations so have to configure the script
        # wrapper and migration context manually
        for base in base_list:
            # Establish a context to access db values - use the bound engine in case
            # of ephemeral connection (in memory sqlite)
            engine = base.metadata.bind
            context = MigrationContext.configure(engine.connect())
            current_revision = context.get_current_revision()

            # Read local config file to find the current "head" revision
            script = ScriptDirectory.from_config(self.alembic_config)
            head_revision = script.get_current_head()

            if current_revision != head_revision:
                raise SimpleMLError(
                    """Attempting to connect to an outdated schema.
                                    Set the parameter `upgrade=True` in the initialize method
                                    or manually execute `simpleml db upgrade` in a shell"""
                )

    def initialize(self, base_list, upgrade=False, validate=True, **kwargs):
        """
        Initialization method to set up database connection and inject
        session manager

        Raises a SimpleML error if database schema is not up to date

        :param drop_tables: Bool, whether to drop existing tables in database
        :param upgrade: Bool, whether to run an upgrade migration after establishing a connection
        :return: None
        """
        # Standard initialization
        super(AlembicDatabase, self).initialize(base_list, **kwargs)

        # Upgrade schema if necessary
        if upgrade:
            self.upgrade()

        # Assert current db schema is up-to-date
        if validate:
            self.validate_schema_version(base_list)


class Database(AlembicDatabase):
    """
    SimpleML specific configuration to interact with the database
    Defaults to sqlite db in filestore directory
    """

    def __init__(
        self,
        configuration_section=None,
        uri=None,
        database=None,
        username=None,
        password=None,
        drivername=None,
        host=None,
        port=None,
        query=None,
        *args,
        **kwargs,
    ):

        if (
            configuration_section is None
            and uri is None
            and all(
                [
                    i is None
                    for i in (database, username, password, drivername, port, query)
                ]
            )
        ):
            # Fill with env variable values if none are passed directly
            configuration_section = DATABASE_CONF
            uri = DATABASE_URI
            database = DATABASE_NAME
            username = DATABASE_USERNAME
            password = DATABASE_PASSWORD
            drivername = DATABASE_DRIVERNAME
            host = DATABASE_HOST
            port = DATABASE_PORT
            query = DATABASE_QUERY

        if (
            configuration_section is None
            and uri is None
            and all(
                [
                    i is None
                    for i in (database, username, password, drivername, port, query)
                ]
            )
        ):
            # Use default creds for a sqlite database in filestore directory if env variables are also null
            LOGGER.info(
                "No database connection specified, using default SQLite db in {}".format(
                    FILESTORE_DIRECTORY
                )
            )
            uri = "sqlite:///{}".format(join(FILESTORE_DIRECTORY, "SimpleML.db"))

        root_path = dirname(dirname(dirname(realpath(__file__))))
        alembic_filepath = join(root_path, "simpleml/migrations/alembic.ini")
        script_location = ""
        super(Database, self).__init__(
            config=CONFIG,
            alembic_filepath=alembic_filepath,
            script_location=script_location,
            configuration_section=configuration_section,
            uri=uri,
            database=database,
            username=username,
            password=password,
            drivername=drivername,
            host=host,
            port=port,
            query=query,
            *args,
            **kwargs,
        )

    def initialize(self, base_list=None, **kwargs):
        """
        Initialization method to set up database connection and inject
        session manager

        Raises a SimpleML error if database schema is not up to date

        :param drop_tables: Bool, whether to drop existing tables in database
        :param upgrade: Bool, whether to run an upgrade migration after establishing a connection
        :return: None
        """
        if base_list is None:  # Use defaults in project
            base_list = [SimplemlCoreSqlalchemy]

        super(Database, self).initialize(base_list, **kwargs)


class DatasetDatabase(BaseDatabase):
    """
    Hardcoded database mapped to dataset storage metadata
    """

    def initialize(self, base_list=None, **kwargs):
        """
        Initialization method to set up database connection and inject
        session manager

        Raises a SimpleML error if database schema is not up to date

        :param drop_tables: Bool, whether to drop existing tables in database
        :param upgrade: Bool, whether to run an upgrade migration after establishing a connection
        :return: None
        """
        if base_list is None:  # Use defaults in project
            base_list = [DatasetStorageSqlalchemy]

        super(DatasetDatabase, self).initialize(base_list, **kwargs)


class BinaryStorageDatabase(BaseDatabase):
    """
    Hardcoded database mapped to binary storage metadata
    """

    def initialize(self, base_list=None, **kwargs):
        """
        Initialization method to set up database connection and inject
        session manager

        Raises a SimpleML error if database schema is not up to date

        :param drop_tables: Bool, whether to drop existing tables in database
        :param upgrade: Bool, whether to run an upgrade migration after establishing a connection
        :return: None
        """
        if base_list is None:  # Use defaults in project
            base_list = [BinaryStorageSqlalchemy]

        super(BinaryStorageDatabase, self).initialize(base_list, **kwargs)
