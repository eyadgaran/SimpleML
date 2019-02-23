'''
Module to set a reference for the SimpleML home directory

Defaults to user's home directory if no environment variable is set
'''

__author__ = 'Elisha Yadgaran'

import os
from configparser import ConfigParser
import logging

LOGGER = logging.getLogger(__name__)


# Configuration
CONFIGURATION_FILE = os.getenv('SIMPLEML_CONFIGURATION_FILE', None)
if CONFIGURATION_FILE is None:
    LOGGER.debug('Configuration File Environment Variable Not Set (`SIMPLEML_CONFIGURATION_FILE`), using default')
    CONFIGURATION_FILE = os.path.expanduser("~/.simpleml/simpleml.conf")

CONFIG = ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')]})
if os.path.isfile(CONFIGURATION_FILE):
    CONFIG.read(CONFIGURATION_FILE)
else:
    LOGGER.warning('No Configuration File Found, Falling Back to Default Values')

# Config Sections
PATH_SECTION = 'path'
CLOUD_SECTION = 'cloud'

# Filestores
if PATH_SECTION in CONFIG:
    SIMPLEML_DIRECTORY = os.path.expanduser(CONFIG.get(PATH_SECTION, 'home_directory'))
    if not os.path.isdir(SIMPLEML_DIRECTORY):
        LOGGER.error('Invalid Home Directory Specified: {}, using ~/.simpleml'.format(SIMPLEML_DIRECTORY))
        SIMPLEML_DIRECTORY = os.path.expanduser("~/.simpleml")

else:
    LOGGER.debug('Home Directory Path Not Set (`[path]`), using default')
    LOGGER.warning('Expected Configuration Section as Follows:')
    LOGGER.warning('[path]')
    LOGGER.warning('home_directory = ~/.simpleml')
    SIMPLEML_DIRECTORY = os.path.expanduser("~/.simpleml")

FILESTORE_DIRECTORY = os.path.join(SIMPLEML_DIRECTORY, 'filestore/')
PICKLED_FILESTORE_DIRECTORY = os.path.join(FILESTORE_DIRECTORY, 'pickle/')
HDF5_FILESTORE_DIRECTORY = os.path.join(FILESTORE_DIRECTORY, 'HDF5/')


# Create Paths if they don't exist
if not os.path.exists(SIMPLEML_DIRECTORY):
    os.makedirs(SIMPLEML_DIRECTORY)

if not os.path.exists(FILESTORE_DIRECTORY):
    os.makedirs(FILESTORE_DIRECTORY)

if not os.path.exists(PICKLED_FILESTORE_DIRECTORY):
    os.makedirs(PICKLED_FILESTORE_DIRECTORY)

if not os.path.exists(HDF5_FILESTORE_DIRECTORY):
    os.makedirs(HDF5_FILESTORE_DIRECTORY)
