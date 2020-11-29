Database
========

Initializing
------------


Environment Variables
---------------------
The full list of variables that can be referenced are::

    - SIMPLEML_CONFIGURATION_FILE
    - SIMPLEML_DATABASE_NAME
    - SIMPLEML_DATABASE_USERNAME
    - SIMPLEML_DATABASE_PASSWORD
    - SIMPLEML_DATABASE_HOST
    - SIMPLEML_DATABASE_PORT
    - SIMPLEML_DATABASE_DRIVERNAME
    - SIMPLEML_DATABASE_QUERY
    - SIMPLEML_DATABASE_CONF
    - SIMPLEML_DATABASE_URI

Code Defaults
-------------
Defaults are specified for expected database parameters::

    - SIMPLEML_CONFIGURATION_FILE = ~/.simpleml/simpleml.conf
    - SIMPLEML_DATABASE_NAME = None
    - SIMPLEML_DATABASE_USERNAME = None
    - SIMPLEML_DATABASE_PASSWORD = None
    - SIMPLEML_DATABASE_HOST = None
    - SIMPLEML_DATABASE_PORT = None
    - SIMPLEML_DATABASE_DRIVERNAME = None
    - SIMPLEML_DATABASE_QUERY = None
    - SIMPLEML_DATABASE_CONF = None
    - SIMPLEML_DATABASE_URI = None


The first is the location of the configuration file. The remainder are database
initialization defaults -- used only if a database class is initialized without
the particular parameters.




Migrations
----------



Optional Schemas
----------------
