"""
Module to set a reference for the SimpleML home directory

Defaults to user's home directory if no environment variable is set

example config file:

```
    [path]
    home_directory = ~/.simpleml  <-- This details where all binaries are stored on the local disk. Only necessary if different than the default

    [libcloud]  <-- This section is used for any persistable that specifies an apache-libcloud persistence location
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

    [s3]
    param = value --> normal key:value syntax. match these to however they are referenced later, examples:
    key = abc123
    secret = superSecure
    region = us-east-1
    something_specific_to_s3 = s3_parameter
    --- Below are internally referenced SimpleML params ---
    driver = S3 --> this must be the Apache Libcloud provider (https://github.com/apache/libcloud/blob/trunk/libcloud/storage/types.py)
    connection_params = key,secret,region,something_specific_to_s3 --> this determines the key: value params passed to the constructor (it can be different for each provider)
    path = simpleml/specific/root --> similar to disk based home directory, cloud home directory will start relative to here
    container = simpleml --> the cloud bucket or container name

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
```
"""

__author__ = "Elisha Yadgaran"

import errno  # os.errno deprecated in python 3.7+
import logging
import os
import tempfile
from configparser import ConfigParser

from simpleml.registries import FILEPATH_REGISTRY

LOGGER = logging.getLogger(__name__)


# Configuration
CONFIGURATION_FILE = os.getenv("SIMPLEML_CONFIGURATION_FILE", None)
if CONFIGURATION_FILE is None:
    LOGGER.debug(
        "Configuration File Environment Variable Not Set (`SIMPLEML_CONFIGURATION_FILE`), using default"
    )
    CONFIGURATION_FILE = os.path.expanduser("~/.simpleml/simpleml.conf")

CONFIG = ConfigParser(converters={"list": lambda x: [i.strip() for i in x.split(",")]})
if os.path.isfile(CONFIGURATION_FILE):
    CONFIG.read(CONFIGURATION_FILE)
else:
    LOGGER.warning("No Configuration File Found, Falling Back to Default Values")

# Config Sections
PATH_SECTION = "path"
LIBCLOUD_SECTION = "libcloud"

# Local Filestore
if PATH_SECTION in CONFIG:
    SIMPLEML_DIRECTORY = os.path.expanduser(CONFIG.get(PATH_SECTION, "home_directory"))
    if not os.path.isdir(SIMPLEML_DIRECTORY):
        LOGGER.error(
            "Invalid Home Directory Specified: {}, using ~/.simpleml".format(
                SIMPLEML_DIRECTORY
            )
        )
        SIMPLEML_DIRECTORY = os.path.expanduser("~/.simpleml")

else:
    LOGGER.debug("Home Directory Path Not Set (`[path]`), using default")
    LOGGER.debug("Expected Configuration Section as Follows:")
    LOGGER.debug("[path]")
    LOGGER.debug("home_directory = ~/.simpleml")
    SIMPLEML_DIRECTORY = os.path.expanduser("~/.simpleml")

# Libcloud configs
if LIBCLOUD_SECTION in CONFIG:
    LIBCLOUD_CONFIG_SECTION = CONFIG.get(LIBCLOUD_SECTION, "section")
    LIBCLOUD_ROOT_PATH = CONFIG.get(LIBCLOUD_CONFIG_SECTION, "path", fallback="")
else:
    LOGGER.debug(
        "Libcloud config parameters not set. Attempts to use persistence patterns with the library will fail"
    )
    LIBCLOUD_ROOT_PATH = ""
    LIBCLOUD_CONFIG_SECTION = None


# Reference paths
PICKLE_DIRECTORY = "pickle/"
HDF5_DIRECTORY = "HDF5/"
PARQUET_DIRECTORY = "parquet/"
CSV_DIRECTORY = "csv/"
ORC_DIRECTORY = "orc/"
JSON_DIRECTORY = "json/"
TENSORFLOW_SAVED_MODEL_DIRECTORY = "saved_model/"
FILESTORE_DIRECTORY = os.path.join(SIMPLEML_DIRECTORY, "filestore/")
SYSTEM_TEMP_DIRECTORY = tempfile.gettempdir()


# register paths for consistent reference
FILEPATH_REGISTRY.register("filestore", FILESTORE_DIRECTORY)
FILEPATH_REGISTRY.register("system_temp", SYSTEM_TEMP_DIRECTORY)
FILEPATH_REGISTRY.register("libcloud_root_path", LIBCLOUD_ROOT_PATH)


# Create Paths if they don't exist - use try/excepts to catch race conditions
def safe_makedirs(dir):
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


if not os.path.exists(SIMPLEML_DIRECTORY):
    safe_makedirs(SIMPLEML_DIRECTORY)

if not os.path.exists(FILESTORE_DIRECTORY):
    safe_makedirs(FILESTORE_DIRECTORY)
