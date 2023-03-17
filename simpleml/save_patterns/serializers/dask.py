"""
Module for Dask save patterns
"""

__author__ = "Elisha Yadgaran"


import glob
import json
from os import makedirs
from os.path import dirname, isfile, join
from typing import Any, Dict, List, Optional, Union

from simpleml.imports import db, dbBag, dd, ddDataFrame
from simpleml.registries import FILEPATH_REGISTRY
from simpleml.save_patterns.base import BaseSerializer
from simpleml.utils.configuration import (
    CSV_DIRECTORY,
    HDF5_DIRECTORY,
    JSON_DIRECTORY,
    ORC_DIRECTORY,
    PARQUET_DIRECTORY,
)


class DaskPersistenceMethods(object):
    """
    Base class for internal dask serialization/deserialization options

    Wraps dd.Dataframe methods with sensible defaults
    Uses dask bag alternatives for optimizations (notably for read parallelization
    and memory handling)
    """

    INDEX_COLUMN = "simpleml_index"

    @staticmethod
    def read_text(*args, **kwargs) -> dbBag:
        """
        Dask Bag wrapper to read text and return a bag
        """
        return db.read_text(*args, **kwargs)

    @classmethod
    def read_csv(
        cls, filepaths: List[str], sample_rows: int = 1000, **kwargs
    ) -> ddDataFrame:
        # Automatically handle index (dask cannot read in index) so
        # set based on output format
        df = dd.read_csv(filepaths, sample_rows=sample_rows, **kwargs)
        if cls.INDEX_COLUMN in df.columns:
            df = df.set_index(cls.INDEX_COLUMN)
        return df

    @staticmethod
    def read_parquet(filepath: str, **kwargs) -> ddDataFrame:
        return dd.read_parquet(filepath, **kwargs)

    @staticmethod
    def read_hdf(filepath: str, **kwargs) -> ddDataFrame:
        return dd.read_hdf(filepath, **kwargs)

    @staticmethod
    def read_orc(filepath: str, **kwargs) -> ddDataFrame:
        return dd.read_orc(filepath, **kwargs)

    @classmethod
    def read_json(cls, filepaths: List[str], persist=False, **kwargs) -> ddDataFrame:
        """
        Uses dask bag implementation to optimize read
        :param persist: bool, flag to return a processing future instead of lazy compute later
        """
        # Automatically handle index
        # df = dd.read_json(filepaths, **kwargs)
        df = cls.read_text(filepaths, **kwargs).map(json.loads).to_dataframe()
        if persist:
            df = df.persist()
        if cls.INDEX_COLUMN in df.columns:
            df = df.set_index(cls.INDEX_COLUMN)
        return df

    @staticmethod
    def read_sql_table(**kwargs) -> ddDataFrame:
        return dd.read_sql_table(**kwargs)

    @staticmethod
    def read_table(**kwargs) -> ddDataFrame:
        return dd.read_table(**kwargs)

    @staticmethod
    def read_fwf(**kwargs) -> ddDataFrame:
        return dd.read_fwf(**kwargs)

    @classmethod
    def to_csv(
        cls, df: ddDataFrame, filepath: str, overwrite: bool = True, **kwargs
    ) -> None:
        if not overwrite:
            # Check if file was already serialized
            if isfile(filepath):
                return
        df.to_csv(filepath, index_label=cls.INDEX_COLUMN, **kwargs)

    @staticmethod
    def to_parquet(
        df: ddDataFrame, filepath: str, overwrite: bool = True, **kwargs
    ) -> None:
        if not overwrite:
            # Check if file was already serialized
            if isfile(filepath):
                return
        df.to_parquet(filepath, **kwargs)

    @staticmethod
    def to_hdf(
        df: ddDataFrame, filepath: str, overwrite: bool = True, **kwargs
    ) -> None:
        if not overwrite:
            # Check if file was already serialized
            if isfile(filepath):
                return
        df.to_hdf(filepath, **kwargs)

    @classmethod
    def to_json(
        cls, df: ddDataFrame, filepath: str, overwrite: bool = True, **kwargs
    ) -> None:
        if not overwrite:
            # Check if file was already serialized
            if isfile(filepath):
                return
        # json records do not include index so artificially inject
        if cls.INDEX_COLUMN in df.columns:
            df.to_json(filepath, **kwargs)
        else:
            df.reset_index(drop=False).rename(
                columns={"index": cls.INDEX_COLUMN}
            ).to_json(filepath, **kwargs)

    @staticmethod
    def to_orc(
        df: ddDataFrame, filepath: str, overwrite: bool = True, **kwargs
    ) -> None:
        if not overwrite:
            # Check if file was already serialized
            if isfile(filepath):
                return
        df.to_orc(filepath, **kwargs)

    @staticmethod
    def to_sql(df: ddDataFrame, **kwargs) -> None:
        df.to_sql(**kwargs)


class DaskParquetSerializer(BaseSerializer):
    @staticmethod
    def serialize(
        obj: ddDataFrame,
        filepath: str,
        format_directory: str = PARQUET_DIRECTORY,
        format_extension: str = ".parquet",
        destination_directory: str = "system_temp",
        **kwargs,
    ) -> Dict[str, str]:
        # Append the filepath to the storage directory
        filepath = join(format_directory, filepath + format_extension)
        full_path = join(FILEPATH_REGISTRY.get(destination_directory), filepath)
        DaskPersistenceMethods.to_parquet(obj, full_path)
        return {"filepath": filepath, "source_directory": destination_directory}

    @staticmethod
    def deserialize(
        filepath: str, source_directory: str = "system_temp", **kwargs
    ) -> Dict[str, Any]:
        full_path = join(FILEPATH_REGISTRY.get(source_directory), filepath)
        return {"obj": DaskPersistenceMethods.read_parquet(full_path)}


class DaskCSVSerializer(BaseSerializer):
    @staticmethod
    def serialize(
        obj: ddDataFrame,
        filepath: str,
        format_directory: str = CSV_DIRECTORY,
        format_extension: str = ".csv",
        destination_directory: str = "system_temp",
        **kwargs,
    ) -> Dict[str, str]:
        # Append the filepath to the storage directory
        # read_csv method expects a * format
        destination_folder = FILEPATH_REGISTRY.get(destination_directory)
        filename_format = join(format_directory, filepath + "-*" + format_extension)
        full_path = join(destination_folder, filename_format)
        DaskPersistenceMethods.to_csv(obj, full_path)
        written_filepaths = glob.glob(full_path)

        # strip out root path to keep relative to directory
        filepaths = []
        for i in written_filepaths:
            relative_path = i.split(destination_folder)[1]
            # strip the preceding /
            if relative_path[0] == "/":
                relative_path = relative_path[1:]
            filepaths.append(relative_path)

        return {"filepaths": filepaths, "source_directory": destination_directory}

    @staticmethod
    def deserialize(
        filepaths: List[str], source_directory: str = "system_temp", **kwargs
    ) -> Dict[str, Any]:
        full_paths = [
            join(FILEPATH_REGISTRY.get(source_directory), filepath)
            for filepath in filepaths
        ]
        return {"obj": DaskPersistenceMethods.read_csv(full_paths)}


class DaskJSONSerializer(BaseSerializer):
    @staticmethod
    def serialize(
        obj: ddDataFrame,
        filepath: str,
        format_directory: str = JSON_DIRECTORY,
        format_extension: str = ".jsonl",
        destination_directory: str = "system_temp",
        **kwargs,
    ) -> Dict[str, str]:
        # Append the filepath to the storage directory
        # read_json method expects a * format
        destination_folder = FILEPATH_REGISTRY.get(destination_directory)
        filename_format = join(format_directory, filepath + "-*" + format_extension)
        full_path = join(destination_folder, filename_format)
        DaskPersistenceMethods.to_json(obj, full_path)

        written_filepaths = glob.glob(full_path)

        # strip out root path to keep relative to directory
        filepaths = []
        for i in written_filepaths:
            relative_path = i.split(destination_folder)[1]
            # strip the preceding /
            if relative_path[0] == "/":
                relative_path = relative_path[1:]
            filepaths.append(relative_path)

        return {"filepaths": filepaths, "source_directory": destination_directory}

    @staticmethod
    def deserialize(
        filepaths: List[str], source_directory: str = "system_temp", **kwargs
    ) -> Dict[str, Any]:
        full_paths = [
            join(FILEPATH_REGISTRY.get(source_directory), filepath)
            for filepath in filepaths
        ]
        return {"obj": DaskPersistenceMethods.read_json(full_paths)}


class DaskHDFSerializer(BaseSerializer):
    @staticmethod
    def serialize(
        obj: ddDataFrame,
        filepath: str,
        format_directory: str = HDF5_DIRECTORY,
        format_extension: str = ".hdf",
        destination_directory: str = "system_temp",
        **kwargs,
    ) -> Dict[str, str]:
        # Append the filepath to the storage directory
        filepath = join(format_directory, filepath + format_extension)
        full_path = join(FILEPATH_REGISTRY.get(destination_directory), filepath)
        DaskPersistenceMethods.to_hdf(obj, full_path)
        return {"filepath": filepath, "source_directory": destination_directory}

    @staticmethod
    def deserialize(
        filepath: str, source_directory: str = "system_temp", **kwargs
    ) -> Dict[str, Any]:
        full_path = join(FILEPATH_REGISTRY.get(source_directory), filepath)
        return {"obj": DaskPersistenceMethods.read_hdf(full_path)}


class DaskORCSerializer(BaseSerializer):
    @staticmethod
    def serialize(
        obj: ddDataFrame,
        filepath: str,
        format_directory: str = ORC_DIRECTORY,
        format_extension: str = ".orc",
        destination_directory: str = "system_temp",
        **kwargs,
    ) -> Dict[str, str]:
        # Append the filepath to the storage directory
        filepath = join(format_directory, filepath + format_extension)
        full_path = join(FILEPATH_REGISTRY.get(destination_directory), filepath)
        DaskPersistenceMethods.to_orc(obj, full_path)
        return {"filepath": filepath, "source_directory": destination_directory}

    @staticmethod
    def deserialize(
        filepath: str, source_directory: str = "system_temp", **kwargs
    ) -> Dict[str, Any]:
        full_path = join(FILEPATH_REGISTRY.get(source_directory), filepath)
        return {"obj": DaskPersistenceMethods.read_orc(full_path)}
