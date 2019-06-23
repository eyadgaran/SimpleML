Configuration
=============

SimpleML contains a series of fallback locations for defining configuration
parameters.

By default, it will start with the ``simpleml.conf`` file in the
library folder (with the exception of the folder path, since it can't read that
before knowing where to look...).

The next location, if the config file does not
contain a value, is through an environment variable.

Finally, it will default
to hard-coded values configured as default class parameters.

To recap::

    1) Configuration File (`simpleml.conf`)
    2) Environment Variable (`SIMPLEML_{parameter}`)
    3) Code Defaults (Check class definitions)


Configuration File
------------------
The configuration file is designed as follows::

    [path]
    home_directory = ~/.simpleml

    [cloud]
    section = gcp-read-only

    [onedrive]
    client_secret = aaaaaabbbbbbbbbcccccc
    root_id = xxxxxyyyyyyzzzzzzzz
    client_id = abcdefg-hijk-lmno-pqrs-tuvwxyz
    scopes = onedrive.readwrite
    redirect_uri = http://localhost:8000/example/callback

    [gcp-read-write]
    driver = GOOGLE_STORAGE
    connection_params = key,secret
    key = read-write@iam.gserviceaccount.com
    secret = ./gcp-read-write.json
    container = simpleml

    [gcp-read-only]
    driver = GOOGLE_STORAGE
    connection_params = key,secret
    key = read-only@iam.gserviceaccount.com
    secret = ./gcp-read-only.json
    container = simpleml

    [simpleml-database]
    database=SimpleML
    username=simpleml
    password=simpleml
    drivername=postgresql
    host=localhost
    port=5432

    [app-database]
    database=APPLICATION_DB
    username=simpleml
    password=simpleml
    drivername=postgresql
    host=localhost
    port=5432


***Note: python configparser interprets `%` as a special character for interpolation.
Add a second, like `%%` to escape the literal value***


Only the sections that are used are necessary. Don't include unnecessary sections in your
config to minimize security exposure if they leak! This entire file would be unnecessary if
cloud storage is not being used, default filepaths are used, and the database connections are initialized by a different
means than `configuration_section`. Breaking down this particular example::

    [cloud]  <-- This section is used for any persistable that specifies a cloud persistence location
    section = gcp-read-only  <-- The name of the heading for the cloud credentials

    [onedrive]  <--- This section outlines an example authorization scheme with onedrive personal
    client_secret = aaaaaabbbbbbbbbcccccc  <--- Put your client secret here
    root_id = xxxxxyyyyyyzzzzzzzz  <--- Put the item id of the root filestore bucket here
    client_id = abcdefg-hijk-lmno-pqrs-tuvwxyz  <--- Put your client_id here
    scopes = onedrive.readwrite  <--- Mark the scopes here (reference the onedrive api for examples)
    redirect_uri = http://localhost:8000/example/callback  <--- Put the callback url here to return the auth token

    [gcp-read-write]  <--- This section outlines an example for a read/write iam in GCP
    driver = GOOGLE_STORAGE  <--- Apache-libcloud driver used (can be any of the supported ones)
    connection_params = key,secret  <--- Which parameters in this section to pass to apache-libcloud
    key = read-write@iam.gserviceaccount.com  <--- The gcp iam account
    secret = ./gcp-read-write.json  <--- The token for that gcp account
    container = simpleml  <--- The gcp container (or "bucket") that houses the files

    [gcp-read-only]  <--- Duplicate example with a read only IAM -- recommended practice to train with the cloud section = gcp-read-write and deploy in production with read only access
    driver = GOOGLE_STORAGE
    connection_params = key,secret
    key = read-only@iam.gserviceaccount.com
    secret = ./gcp-read-only.json
    container = simpleml

    [simpleml-database]  <--- Database credentials for the simpleml models (used by specifying Database(configuration_section='simpleml-database'))
    database=SimpleML
    username=simpleml
    password=simpleml
    drivername=postgresql
    host=localhost
    port=5432

    [app-database]  <--- Database credentials for application logs (used by specifying Database(configuration_section='app-database'))
    database=APPLICATION_DB
    username=simpleml
    password=simpleml
    drivername=postgresql
    host=localhost
    port=5432


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
    - SIMPLEML_DATABASE_CONF
    - SIMPLEML_DATABASE_URI

Code Defaults
-------------
Defaults are specified for expected database parameters::

    - SIMPLEML_CONFIGURATION_FILE = ~/.simpleml/simpleml.conf
    - SIMPLEML_DATABASE_NAME = SimpleML
    - SIMPLEML_DATABASE_USERNAME = simpleml
    - SIMPLEML_DATABASE_PASSWORD = simpleml
    - SIMPLEML_DATABASE_HOST = localhost
    - SIMPLEML_DATABASE_PORT = 5432
    - SIMPLEML_DATABASE_DRIVERNAME = postgresql
    - SIMPLEML_DATABASE_CONF = None
    - SIMPLEML_DATABASE_URI = None


The first is the location of the configuration file. The remainder are database
initialization defaults -- used only if a database class is initialized without
the particular parameters.
