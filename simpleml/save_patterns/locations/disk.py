"""
Module for Disk based locations
"""

__author__ = "Elisha Yadgaran"


import shutil
from os import makedirs
from os.path import dirname, isfile, join
from typing import Any, Dict, List, Optional

from simpleml.registries import FILEPATH_REGISTRY
from simpleml.save_patterns.base import BaseSerializer


class DiskIOMethods(object):
    @staticmethod
    def copy_file(src: str, destination: str) -> None:
        # safety check for the destination path
        makedirs(dirname(destination), exist_ok=True)
        shutil.copy2(src, destination)

    @staticmethod
    def copy_directory(src: str, destination: str) -> None:
        # safety check for the destination path
        makedirs(dirname(destination), exist_ok=True)
        shutil.copytree(src, destination)


class FilestoreCopyFileLocation(BaseSerializer):
    @staticmethod
    def serialize(filepath: str,
                  source_directory: str = 'system_temp',
                  destination_directory: str = 'filestore', **kwargs) -> Dict[str, str]:
        DiskIOMethods.copy_file(
            join(FILEPATH_REGISTRY.get(source_directory), filepath),
            join(FILEPATH_REGISTRY.get(destination_directory), filepath),
        )

        return {"filepath": filepath, "source_directory": destination_directory}

    @staticmethod
    def deserialize(filepath: str,
                    source_directory: str = 'filestore',
                    destination_directory: str = 'system_temp', **kwargs) -> Dict[str, str]:
        DiskIOMethods.copy_file(
            join(FILEPATH_REGISTRY.get(source_directory), filepath),
            join(FILEPATH_REGISTRY.get(destination_directory), filepath),
        )

        return {"filepath": filepath, "source_directory": destination_directory}


class FilestoreCopyFilesLocation(BaseSerializer):
    @staticmethod
    def serialize(filepaths: List[str],
                  source_directory: str = 'system_temp',
                  destination_directory: str = 'filestore', **kwargs) -> Dict[str, str]:

        for filepath in filepaths:
            DiskIOMethods.copy_file(
                join(FILEPATH_REGISTRY.get(source_directory), filepath),
                join(FILEPATH_REGISTRY.get(destination_directory), filepath),
            )

        return {"filepaths": filepaths, "source_directory": destination_directory}

    @staticmethod
    def deserialize(filepaths: List[str],
                    source_directory: str = 'filestore',
                    destination_directory: str = 'system_temp', **kwargs) -> Dict[str, str]:

        for filepath in filepaths:
            DiskIOMethods.copy_file(
                join(FILEPATH_REGISTRY.get(source_directory), filepath),
                join(FILEPATH_REGISTRY.get(destination_directory), filepath),
            )

        return {"filepaths": filepaths, "source_directory": destination_directory}


class FilestoreCopyFolderLocation(BaseSerializer):
    @staticmethod
    def serialize(filepath: str,
                  source_directory: str = 'system_temp',
                  destination_directory: str = 'filestore', **kwargs) -> Dict[str, str]:
        DiskIOMethods.copy_directory(
            join(FILEPATH_REGISTRY.get(source_directory), filepath),
            join(FILEPATH_REGISTRY.get(destination_directory), filepath),
        )

        return {"filepath": filepath, "source_directory": destination_directory}

    @staticmethod
    def deserialize(filepath: str,
                    source_directory: str = 'filestore',
                    destination_directory: str = 'system_temp', **kwargs) -> Dict[str, str]:
        DiskIOMethods.copy_directory(
            join(FILEPATH_REGISTRY.get(source_directory), filepath),
            join(FILEPATH_REGISTRY.get(destination_directory), filepath),
        )

        return {"filepath": filepath, "source_directory": destination_directory}


class FilestorePassthroughLocation(BaseSerializer):
    @staticmethod
    def serialize(**kwargs) -> Dict[str, str]:
        return kwargs

    @staticmethod
    def deserialize(**kwargs) -> Dict[str, str]:
        return kwargs
