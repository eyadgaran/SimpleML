'''
Package for the different save patterns available
Patterns are loaded into global registry on import and more can be added
externally by decorating

Nomenclature -> Save Location : Save Format

- Database Storage
    - database_table: Dataframe saving (as tables in dedicated schema)
    - database_pickled: In database as a binary blob
    - database_hdf5: In database as a binary blob
- Local Filesystem Storage
    - disk_pickled: Pickled file on local disk
    - disk_hdf5: HDF5 file on local disk
    - disk_keras_hdf5: Keras formatted HDF5 file on local disk
- Cloud Storage
    - cloud_pickled: Pickled file on cloud backend
    - cloud_hdf5: HDF5 file on cloud backend
    - cloud_keras_hdf5: Keras formatted HDF5 file on cloud backend
  Supported Backends:
    - Amazon S3
    - Google Cloud Platform
    - Microsoft Azure
    - Microsoft Onedrive
    - Aurora
    - Backblaze B2
    - DigitalOcean Spaces
    - OpenStack Swift
  Backend is determined by `cloud_section` in the configuration file
- Remote filestore saving
    - SCP to remote server
'''


__author__ = 'Elisha Yadgaran'


# Auto import all submodules to ensure registration on library import
from .base import SavePatternMixin, BaseSavePattern
from .decorators import SavePatternDecorators, register_save_pattern, deregister_save_pattern

from . import database
from . import libcloud
from . import local
from . import onedrive
