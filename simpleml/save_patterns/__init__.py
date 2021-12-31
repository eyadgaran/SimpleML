'''
Package for artifact persistence. Bindings are automatically included for SimpleML
persistables, but patterns can be used for any objects or frameworks.

Patterns are loaded into global registry on import and more can be added
externally by decorating

Patterns can be named anything since they are only mappings in the registry.
Convention is -> Location : Serializer : Format(s)



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


import logging
from os.path import join
from typing import Any, Optional

from simpleml.utils.configuration import PICKLE_DIRECTORY

# Auto import all submodules to ensure registration on library import
from .base import BaseSavePattern, BaseSerializer
from .decorators import (SavePatternDecorators, deregister_save_pattern,
                         register_save_pattern)
from .locations.disk import (FilestoreCopyFileLocation,
                             FilestoreCopyFilesLocation,
                             FilestoreCopyFolderLocation,
                             FilestorePassthroughLocation)
from .locations.libcloud import (LibcloudCopyFileLocation,
                                 LibcloudCopyFilesLocation,
                                 LibcloudCopyFolderLocation)
from .serializers.cloudpickle import CloudpickleFileSerializer
from .serializers.dask import DaskCSVSerializer, DaskParquetSerializer

LOGGER = logging.getLogger(__name__)


'''
(Cloud)Pickle Save Patterns
'''



'''
Dask Save Patterns
'''


'''
Pandas Save Patterns
'''

'''
Keras Save Patterns
'''

'''
Hickle Save Patterns
'''

'''
Database Save Patterns
'''
