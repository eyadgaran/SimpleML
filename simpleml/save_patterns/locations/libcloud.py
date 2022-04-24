"""
Module for cloud save pattern definitions
Uses Apache Libcloud as a universal engine
"""

__author__ = "Elisha Yadgaran"


from os import makedirs, walk
from os.path import dirname, isdir, isfile, join
from typing import Any, Dict, List

from simpleml.imports import Provider, get_driver
from simpleml.registries import FILEPATH_REGISTRY
from simpleml.save_patterns.base import BaseSerializer
from simpleml.utils.configuration import CONFIG, LIBCLOUD_CONFIG_SECTION


class LibcloudMethods(object):
    """
    Mixin class to save/load objects via Apache Libcloud

    Generic api for all cloud providers so naming convention is extremely important
    to follow in the config. Please reference libcloud documentation for supported
    input parameters

    ```
    [cloud]
    section = `name of the config section to use, ex: s3`

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
    ```

    How this gets used:
    ```
    from libcloud.storage.types import Provider
    from libcloud.storage.providers import get_driver

    cloud_section = CONFIG.get(CLOUD_SECTION, 'section')
    connection_params = CONFIG.getlist(cloud_section, 'connection_params')
    root_path = CONFIG.get(cloud_section, 'path', fallback='')

    driver_cls = get_driver(getattr(Provider, CONFIG.get(cloud_section, 'driver')))
    driver = driver_cls(**{param: CONFIG.get(cloud_section, param) for param in connection_params})
    container = driver.get_container(container_name=CONFIG.get(cloud_section, 'container'))
    extra = {'content_type': 'application/octet-stream'}

    obj = driver.upload_object(LOCAL_FILE_PATH,
                               container=container,
                               object_name=root_path + simpleml_folder_path + filename,
                               extra=extra)

    obj = driver.download_object(CLOUD_OBJECT,
                                 destination_path=LOCAL_FILE_PATH,
                                 overwrite_existing=True,
                                 delete_on_failure=True)
    ```
    """

    @staticmethod
    def get_driver_config(config_section: str = None, **kwargs) -> Dict[str, str]:
        if config_section is None:
            # use default
            config_section = LIBCLOUD_CONFIG_SECTION
        connection_params = CONFIG.getlist(config_section, "connection_params")
        return {param: CONFIG.get(config_section, param) for param in connection_params}

    @classmethod
    def get_driver(cls, provider: str = None, **kwargs) -> Any:
        if provider is None:
            provider = CONFIG.get(LIBCLOUD_CONFIG_SECTION, "driver")
        driver_cls = get_driver(getattr(Provider, provider))
        driver = driver_cls(**cls.get_driver_config(**kwargs))
        return driver

    @staticmethod
    def get_container_name(config_section: str = None, **kwargs) -> str:
        if config_section is None:
            # use default
            config_section = LIBCLOUD_CONFIG_SECTION
        return CONFIG.get(config_section, "container")

    @staticmethod
    def upload(
        driver, source_filepath: str, destination_filepath: str, container_name: str
    ) -> None:
        """
        Upload any file from disk to cloud
        """
        container = driver.get_container(container_name=container_name)
        extra = {"content_type": "application/octet-stream"}
        driver.upload_object(
            source_filepath,
            container=container,
            object_name=destination_filepath,
            extra=extra,
        )

    @staticmethod
    def download(
        driver, source_filepath: str, destination_filepath: str, container_name: str
    ) -> None:
        """
        Download any file from cloud to disk
        """
        obj = driver.get_object(
            container_name=container_name, object_name=source_filepath
        )

        driver.download_object(
            obj,
            destination_path=destination_filepath,
            overwrite_existing=True,
            delete_on_failure=True,
        )


class LibcloudCopyFileLocation(BaseSerializer):
    @staticmethod
    def serialize(
        filepath: str,
        source_directory: str = "system_temp",
        destination_directory: str = "libcloud_root_path",
        **kwargs,
    ) -> Dict[str, str]:

        source_filepath = join(FILEPATH_REGISTRY.get(source_directory), filepath)
        if isdir(source_filepath):
            raise ValueError(
                "Cannot use file persistence pattern for folders. Use `LibcloudCopyFolderLocation` instead"
            )

        LibcloudMethods.upload(
            LibcloudMethods.get_driver(**kwargs),
            source_filepath,
            join(FILEPATH_REGISTRY.get(destination_directory), filepath),
            LibcloudMethods.get_container_name(**kwargs),
        )

        return {"filepath": filepath, "source_directory": destination_directory}

    @staticmethod
    def deserialize(
        filepath: str,
        source_directory: str = "libcloud_root_path",
        destination_directory: str = "system_temp",
        **kwargs,
    ) -> Dict[str, str]:
        LibcloudMethods.download(
            LibcloudMethods.get_driver(**kwargs),
            join(FILEPATH_REGISTRY.get(source_directory), filepath),
            join(FILEPATH_REGISTRY.get(destination_directory), filepath),
            LibcloudMethods.get_container_name(**kwargs),
        )

        return {"filepath": filepath, "source_directory": destination_directory}


class LibcloudCopyFolderLocation(BaseSerializer):
    """
    Libcloud doesnt have a notion of folder objects so iterate through filepaths
    individually
    """

    @staticmethod
    def serialize(
        filepath: str,
        source_directory: str = "system_temp",
        destination_directory: str = "libcloud_root_path",
        **kwargs,
    ) -> Dict[str, str]:
        source_folder = FILEPATH_REGISTRY.get(source_directory)
        destination_folder = FILEPATH_REGISTRY.get(destination_directory)
        driver = LibcloudMethods.get_driver(**kwargs)
        container = LibcloudMethods.get_container_name(**kwargs)

        if not isdir(join(source_folder, filepath)):
            raise ValueError(
                "Cannot use folder persistence pattern for files. Use `LibcloudCopyFileLocation` instead"
            )

        # walkthrough all subpaths
        filepaths = []
        for (dirpath, dirnames, filenames) in walk(join(source_folder, filepath)):
            for filename in filenames:
                # strip out root path to keep relative to directory
                filename = join(dirpath, filename).split(source_folder)[1]
                # strip the preceding /
                if filename[0] == "/":
                    filename = filename[1:]
                filepaths.append(filename)

        for file in filepaths:
            source_filepath = join(source_folder, file)
            destination_filepath = join(destination_folder, file)
            LibcloudMethods.upload(
                driver, source_filepath, destination_filepath, container
            )

        return {"filepaths": filepaths, "source_directory": destination_directory}

    @staticmethod
    def common_path(paths: List[str]) -> str:
        """
        Helper utility to return the common parent path for a bunch of filepaths
        """
        split_paths = [i.split("/") for i in paths]
        common_splits = []
        shortest_split = min([len(i) for i in split_paths])

        index = 0
        while index < shortest_split:
            parent_paths = [i[index] for i in split_paths]
            if len(set(parent_paths)) > 1:
                break
            common_splits.append(parent_paths[0])
            index += 1

        return join(*common_splits)

    @classmethod
    def deserialize(
        cls,
        filepaths: List[str],
        source_directory: str = "libcloud_root_path",
        destination_directory: str = "system_temp",
        **kwargs,
    ) -> Dict[str, str]:
        driver = LibcloudMethods.get_driver(**kwargs)
        container = LibcloudMethods.get_container_name(**kwargs)

        for file in filepaths:
            source_filepath = join(FILEPATH_REGISTRY.get(source_directory), file)
            destination_filepath = join(
                FILEPATH_REGISTRY.get(destination_directory), file
            )
            # safety check for the destination path
            makedirs(dirname(destination_filepath), exist_ok=True)
            LibcloudMethods.download(
                driver, source_filepath, destination_filepath, container
            )

        folder_filepath = cls.common_path(filepaths)

        return {"filepath": folder_filepath, "source_directory": destination_directory}


class LibcloudCopyFilesLocation(BaseSerializer):
    """
    Libcloud transport for many individual files
    """

    @staticmethod
    def serialize(
        filepaths: List[str],
        source_directory: str = "system_temp",
        destination_directory: str = "libcloud_root_path",
        **kwargs,
    ) -> Dict[str, str]:
        source_folder = FILEPATH_REGISTRY.get(source_directory)
        destination_folder = FILEPATH_REGISTRY.get(destination_directory)
        driver = LibcloudMethods.get_driver(**kwargs)
        container = LibcloudMethods.get_container_name(**kwargs)

        for file in filepaths:
            source_filepath = join(source_folder, file)
            destination_filepath = join(destination_folder, file)

            if isdir(source_filepath):
                raise ValueError(
                    "Cannot use file persistence pattern for folder. Use `LibcloudCopyFolderLocation` instead"
                )

            LibcloudMethods.upload(
                driver, source_filepath, destination_filepath, container
            )

        return {"filepaths": filepaths, "source_directory": destination_directory}

    @classmethod
    def deserialize(
        cls,
        filepaths: List[str],
        source_directory: str = "libcloud_root_path",
        destination_directory: str = "system_temp",
        **kwargs,
    ) -> Dict[str, str]:
        driver = LibcloudMethods.get_driver(**kwargs)
        container = LibcloudMethods.get_container_name(**kwargs)

        for file in filepaths:
            source_filepath = join(FILEPATH_REGISTRY.get(source_directory), file)
            destination_filepath = join(
                FILEPATH_REGISTRY.get(destination_directory), file
            )
            # safety check for the destination path
            makedirs(dirname(destination_filepath), exist_ok=True)
            LibcloudMethods.download(
                driver, source_filepath, destination_filepath, container
            )

        return {"filepaths": filepaths, "source_directory": destination_directory}
