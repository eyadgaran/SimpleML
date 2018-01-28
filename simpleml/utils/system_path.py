import os

SIMPLEML_DIRECTORY = os.getenv('SIMPLEML_DIRECTORY_PATH', os.path.expanduser("~/.simpleml"))
FILESTORE_DIRECTORY = os.path.join(SIMPLEML_DIRECTORY, 'filestore/')

if not os.path.exists(SIMPLEML_DIRECTORY):
    os.makedirs(SIMPLEML_DIRECTORY)

if not os.path.exists(FILESTORE_DIRECTORY):
    os.makedirs(FILESTORE_DIRECTORY)
