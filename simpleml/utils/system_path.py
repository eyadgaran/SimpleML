'''
Module to set a reference for the SimpleML home directory

Defaults to user's home directory if no environment variable is set
'''

__author__ = 'Elisha Yadgaran'

import os

# Base Directory
SIMPLEML_DIRECTORY = os.getenv('SIMPLEML_DIRECTORY_PATH', os.path.expanduser("~/.simpleml"))

# Filestores
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
