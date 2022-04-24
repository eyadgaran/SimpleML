"""
File based datasets
"""

__author__ = "Elisha Yadgaran"


from typing import Dict, Optional

from simpleml.save_patterns.serializers.dask import DaskPersistenceMethods
from simpleml.utils.errors import DatasetError

from .base import BaseDaskDataset

DASK_READER_MAP = {
    "csv": DaskPersistenceMethods.read_csv,
    "json": DaskPersistenceMethods.read_json,
    "parquet": DaskPersistenceMethods.read_parquet,
}


class DaskFileBasedDataset(BaseDaskDataset):
    """
    Dask dataset class that generates the dataframe by reading in a file
    """

    def __init__(
        self, filepath: str, format: str, reader_params: Optional[Dict] = None, **kwargs
    ):
        super().__init__(**kwargs)
        if format not in DASK_READER_MAP:
            raise DatasetError(
                f"No reader configured for provided file format: {format}"
            )
        self.config.update(
            {
                "filepath": filepath,
                "format": format,
                "reader_params": reader_params or {},
            }
        )

    def build_dataframe(self) -> None:
        filepath = self.config.get("filepath")
        format = self.config.get("format")
        params = self.config.get("reader_params")
        self.dataframe = DASK_READER_MAP[format](filepath, **params)
