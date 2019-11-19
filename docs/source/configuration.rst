Configuration
=============

SimpleML contains a series of fallback locations for defining configuration
parameters. That means: no configuration is necessary to get started with the
default implementation. For most users starting out that is good enough, but
for advanced users and production workloads, it is recommended to configure a
production-grade stack.

The first configuration that is looked for is the environment variable indicating
the location of the main configuration file, ``SIMPLEML_CONFIGURATION_FILE``.
This variable, when set, should include a filepath to the configuration file (configparser
compatible format). If that is not set, the default location is used,
``~/.simpleml/simpleml.conf``.

If the configuration file is found, the next configuration looked for is the
home directory to store all SimpleML related files. This includes the default
database as well as any binaries of trained persistables. If the configuration
file is not found, or the path is not specified inside it, the default location
will be used: `~/.simpleml`


To recap::

    1) `SIMPLEML_CONFIGURATION_FILE` or default of `~/.simpleml/simpleml.conf`
    2) `[PATH]` section in configuration file or default of `~/.simpleml`


Configuration File
------------------
The configuration file can be used to house a lot more that just the basic
configuration detailed already. The design is meant to extend to an arbitrary
number of sections that can be parsed out for different uses by different classes.
The below example highlights the expected primary sections as well as a number
of optional ones that different modules access to streamline credentials and
configuration. The base design is as follows::

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


Only the sections that are used are necessary. Don't include unnecessary sections in a
config to minimize security exposure if they leak! This entire file would be unnecessary if
cloud storage is not being used, default filepaths are used, and the database connections are initialized by a different
means than `configuration_section`. Breaking down this particular example::

    [path]
    home_directory = ~/.simpleml  <-- This details where all binaries are stored on the local disk. Only necessary if different than the default

    [cloud]  <-- This section is used for any persistable that specifies a cloud persistence location
    section = gcp-read-only  <-- The name of the heading for the cloud credentials.
                                 In this example this line is the only value that needs to be changed to move
                                 from saving in GCP to, say, S3. No code would have to change whatsoever because
                                 the save location is "cloud" which does a lookup in the config

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

    [gcp-read-only]  <--- Duplicate example with a read only IAM -- recommended practice to train with the
                          cloud section = gcp-read-write and deploy in production with read only access
    driver = GOOGLE_STORAGE
    connection_params = key,secret
    key = read-only@iam.gserviceaccount.com
    secret = ./gcp-read-only.json
    container = simpleml

    [simpleml-database]  <--- Database credentials for the simpleml models
                              (used by specifying Database(configuration_section='simpleml-database'))
    database=SimpleML
    username=simpleml
    password=simpleml
    drivername=postgresql
    host=localhost
    port=5432

    [app-database]  <--- Database credentials for application logs (used by
                         specifying Database(configuration_section='app-database'))
    database=APPLICATION_DB
    username=simpleml
    password=simpleml
    drivername=postgresql
    host=localhost
    port=5432


Directory Structure
-------------------
The folder structure inside the home directory will look as follows::

  .simpleml/
  ├── SimpleML.db
  ├── simpleml.conf
  ├── filestore
  │   ├── HDF5
  │   └── pickle

  Where HDF5 and pickle refer to the protocols of the binaries stored within
  them.
