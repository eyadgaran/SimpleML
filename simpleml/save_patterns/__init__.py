"""
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
"""


__author__ = "Elisha Yadgaran"


import logging
from os.path import join
from typing import Any, Optional

from simpleml.utils.configuration import PICKLE_DIRECTORY

# Auto import all submodules to ensure registration on library import
from .base import BaseSavePattern, BaseSerializer
from .decorators import (
    SavePatternDecorators,
    deregister_save_pattern,
    register_save_pattern,
)
from .locations.disk import (
    FilestoreCopyFileLocation,
    FilestoreCopyFilesLocation,
    FilestoreCopyFolderLocation,
    FilestorePassthroughLocation,
)
from .locations.libcloud import (
    LibcloudCopyFileLocation,
    LibcloudCopyFilesLocation,
    LibcloudCopyFolderLocation,
)
from .serializers.cloudpickle import CloudpickleFileSerializer
from .serializers.dask import (
    DaskCSVSerializer,
    DaskJSONSerializer,
    DaskParquetSerializer,
)
from .serializers.keras import KerasH5Serializer, KerasSavedModelSerializer
from .serializers.pandas import (
    PandasCSVSerializer,
    PandasJSONSerializer,
    PandasParquetSerializer,
)

LOGGER = logging.getLogger(__name__)


"""
(Cloud)Pickle Save Patterns
"""


@SavePatternDecorators.register_save_pattern
class CloudpickleDiskSavePattern(BaseSavePattern):
    """
    Save pattern implementation to save objects to disk in pickled format
    """

    SAVE_PATTERN = "disk_pickled"
    serializers = (CloudpickleFileSerializer, FilestoreCopyFileLocation)
    deserializers = (FilestorePassthroughLocation, CloudpickleFileSerializer)

    @classmethod
    def load(cls, legacy: Optional[str] = None, **kwargs):
        """
        Catch for legacy filepath data to dynamically update to new convention
        """
        if legacy is not None:
            # legacy behavior for filename without directory info
            filepath = join(PICKLE_DIRECTORY, legacy)
            source_directory = "filestore"
            LOGGER.debug(
                f"Overwriting legacy filepath param with {filepath} and source_directory with {source_directory}"
            )

            kwargs["filepath"] = filepath
            kwargs["source_directory"] = source_directory

        return super().load(**kwargs)


@SavePatternDecorators.register_save_pattern
class CloudpickleLibcloudSavePattern(BaseSavePattern):
    """
    Save pattern implementation to save objects to disk in pickled format
    """

    SAVE_PATTERN = "cloud_pickled"
    serializers = (CloudpickleFileSerializer, LibcloudCopyFileLocation)
    deserializers = (LibcloudCopyFileLocation, CloudpickleFileSerializer)

    @classmethod
    def load(cls, legacy: Optional[str] = None, **kwargs):
        """
        Catch for legacy filepath data to dynamically update to new convention
        """
        if legacy is not None:
            # legacy behavior for filename without directory info
            filepath = join(PICKLE_DIRECTORY, legacy)
            source_directory = "libcloud_root_path"
            LOGGER.debug(
                f"Overwriting legacy filepath param with {filepath} and source_directory with {source_directory}"
            )

            kwargs["filepath"] = filepath
            kwargs["source_directory"] = source_directory

        return super().load(**kwargs)


"""
Dask Save Patterns
"""


@SavePatternDecorators.register_save_pattern
class DaskDiskParquetSavePattern(BaseSavePattern):
    """
    Save pattern implementation to save dask objects to disk in parquet format
    """

    SAVE_PATTERN = "dask_disk_parquet"
    serializers = (DaskParquetSerializer, FilestoreCopyFolderLocation)
    deserializers = (FilestorePassthroughLocation, DaskParquetSerializer)


@SavePatternDecorators.register_save_pattern
class DaskLibcloudParquetSavePattern(BaseSavePattern):
    """
    Save pattern implementation to save dask objects to cloud via apached-libcloud in parquet format
    """

    SAVE_PATTERN = "dask_libcloud_parquet"
    serializers = (DaskParquetSerializer, LibcloudCopyFolderLocation)
    deserializers = (LibcloudCopyFolderLocation, DaskParquetSerializer)


@SavePatternDecorators.register_save_pattern
class DaskDiskCSVSavePattern(BaseSavePattern):
    """
    Save pattern implementation to save dask objects to disk in csv format
    """

    SAVE_PATTERN = "dask_disk_csv"
    serializers = (DaskCSVSerializer, FilestoreCopyFilesLocation)
    deserializers = (FilestorePassthroughLocation, DaskCSVSerializer)


@SavePatternDecorators.register_save_pattern
class DaskLibcloudCSVSavePattern(BaseSavePattern):
    """
    Save pattern implementation to save dask objects to cloud via apached-libcloud in csv format
    """

    SAVE_PATTERN = "dask_libcloud_csv"
    serializers = (DaskCSVSerializer, LibcloudCopyFilesLocation)
    deserializers = (LibcloudCopyFilesLocation, DaskCSVSerializer)


@SavePatternDecorators.register_save_pattern
class DaskDiskJSONSavePattern(BaseSavePattern):
    """
    Save pattern implementation to save dask objects to disk in json format
    """

    SAVE_PATTERN = "dask_disk_json"
    serializers = (DaskJSONSerializer, FilestoreCopyFilesLocation)
    deserializers = (FilestorePassthroughLocation, DaskJSONSerializer)


@SavePatternDecorators.register_save_pattern
class DaskLibcloudJSONSavePattern(BaseSavePattern):
    """
    Save pattern implementation to save dask objects to cloud via apached-libcloud in json format
    """

    SAVE_PATTERN = "dask_libcloud_json"
    serializers = (DaskJSONSerializer, LibcloudCopyFilesLocation)
    deserializers = (LibcloudCopyFilesLocation, DaskJSONSerializer)


"""
Pandas Save Patterns
"""


@SavePatternDecorators.register_save_pattern
class PandasDiskParquetSavePattern(BaseSavePattern):
    """
    Save pattern implementation to save pandas objects to disk in parquet format
    """

    SAVE_PATTERN = "pandas_disk_parquet"
    serializers = (PandasParquetSerializer, FilestoreCopyFileLocation)
    deserializers = (FilestorePassthroughLocation, PandasParquetSerializer)


@SavePatternDecorators.register_save_pattern
class PandasLibcloudParquetSavePattern(BaseSavePattern):
    """
    Save pattern implementation to save pandas objects to cloud via apached-libcloud in parquet format
    """

    SAVE_PATTERN = "pandas_libcloud_parquet"
    serializers = (PandasParquetSerializer, LibcloudCopyFileLocation)
    deserializers = (LibcloudCopyFileLocation, PandasParquetSerializer)


@SavePatternDecorators.register_save_pattern
class PandasDiskCSVSavePattern(BaseSavePattern):
    """
    Save pattern implementation to save pandas objects to disk in csv format
    """

    SAVE_PATTERN = "pandas_disk_csv"
    serializers = (PandasCSVSerializer, FilestoreCopyFileLocation)
    deserializers = (FilestorePassthroughLocation, PandasCSVSerializer)


@SavePatternDecorators.register_save_pattern
class PandasLibcloudCSVSavePattern(BaseSavePattern):
    """
    Save pattern implementation to save pandas objects to cloud via apached-libcloud in csv format
    """

    SAVE_PATTERN = "pandas_libcloud_csv"
    serializers = (PandasCSVSerializer, LibcloudCopyFileLocation)
    deserializers = (LibcloudCopyFileLocation, PandasCSVSerializer)


@SavePatternDecorators.register_save_pattern
class PandasDiskJSONSavePattern(BaseSavePattern):
    """
    Save pattern implementation to save pandas objects to disk in json format
    """

    SAVE_PATTERN = "pandas_disk_json"
    serializers = (PandasJSONSerializer, FilestoreCopyFileLocation)
    deserializers = (FilestorePassthroughLocation, PandasJSONSerializer)


@SavePatternDecorators.register_save_pattern
class PandasLibcloudJSONSavePattern(BaseSavePattern):
    """
    Save pattern implementation to save pandas objects to cloud via apached-libcloud in json format
    """

    SAVE_PATTERN = "pandas_libcloud_json"
    serializers = (PandasJSONSerializer, LibcloudCopyFileLocation)
    deserializers = (LibcloudCopyFileLocation, PandasJSONSerializer)


"""
Keras Save Patterns
"""


@SavePatternDecorators.register_save_pattern
class KerasDiskSavedModelSavePattern(BaseSavePattern):
    """
    Save pattern implementation to save keras objects to disk in savedModel format
    """

    SAVE_PATTERN = "keras_disk_saved_model"
    serializers = (KerasSavedModelSerializer, FilestoreCopyFolderLocation)
    deserializers = (FilestorePassthroughLocation, KerasSavedModelSerializer)


@SavePatternDecorators.register_save_pattern
class KerasLibcloudSavedModelSavePattern(BaseSavePattern):
    """
    Save pattern implementation to save keras objects to cloud via apached-libcloud in savedModel format
    """

    SAVE_PATTERN = "keras_libcloud_saved_model"
    serializers = (KerasSavedModelSerializer, LibcloudCopyFolderLocation)
    deserializers = (LibcloudCopyFolderLocation, KerasSavedModelSerializer)


@SavePatternDecorators.register_save_pattern
class KerasDiskH5SavePattern(BaseSavePattern):
    """
    Save pattern implementation to save keras objects to disk in h5 format
    """

    SAVE_PATTERN = "keras_disk_h5"
    serializers = (KerasH5Serializer, FilestoreCopyFileLocation)
    deserializers = (FilestorePassthroughLocation, KerasH5Serializer)


@SavePatternDecorators.register_save_pattern
class KerasLibcloudH5SavePattern(BaseSavePattern):
    """
    Save pattern implementation to save keras objects to cloud via apached-libcloud in h5 format
    """

    SAVE_PATTERN = "keras_libcloud_h5"
    serializers = (KerasH5Serializer, LibcloudCopyFileLocation)
    deserializers = (LibcloudCopyFileLocation, KerasH5Serializer)


"""
Hickle Save Patterns
"""

"""
Database Save Patterns
"""

"""
Onedrive Save Patterns
"""
