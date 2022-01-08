'''
Extension implementations for loading file based datasets
'''

__author__ = 'Elisha Yadgaran'


from typing import Dict, Optional

from simpleml.save_patterns.serializers.pandas import PandasPersistenceMethods

from .base import BasePandasDataset

PANDAS_READER_MAP = {
    'csv': PandasPersistenceMethods.read_csv,
    'json': PandasPersistenceMethods.read_json,
    'parquet': PandasPersistenceMethods.read_parquet,
}


class PandasFileBasedDataset(BasePandasDataset):
    '''
    Pandas dataset class that generates the dataframe by reading in a file
    '''

    def __init__(self, filepath: str, format: str, reader_params: Optional[Dict] = None, **kwargs):
        super().__init__(**kwargs)
        if format not in PANDAS_READER_MAP:
            raise DatasetError(f'No reader configured for provided file format: {format}')
        self.config.update({'filepath': filepath, 'format': format, 'reader_params': reader_params or {}})

    def build_dataframe(self) -> None:
        filepath = self.config.get('filepath')
        format = self.config.get('format')
        params = self.config.get('reader_params')
        self.dataframe = PANDAS_READER_MAP[format](filepath, **params)
