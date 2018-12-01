from simpleml.persistables.meta_registry import RawDatasetRegistry
from simpleml.datasets.base_dataset import BaseDataset
from simpleml.persistables.dataset_storage import RawDatasetStorage, RAW_DATASET_SCHEMA
from simpleml.datasets.pandas_mixin import PandasDatasetMixin
from simpleml.datasets.numpy_mixin import NumpyDatasetMixin
from sqlalchemy import UniqueConstraint, Index

__author__ = 'Elisha Yadgaran'


class BaseRawDataset(BaseDataset):
    '''
    Base class for all Raw Dataset objects.

    -------
    Schema
    -------
    No additional columns
    '''

    __tablename__ = 'raw_datasets'
    __metaclass__ = RawDatasetRegistry

    __table_args__ = (
        # Unique constraint for versioning
        UniqueConstraint('name', 'version', name='raw_dataset_name_version_unique'),
        # Index for searching through friendly names
        Index('raw_dataset_name_index', 'name'),
     )

    @property
    def _schema(self):
        return RAW_DATASET_SCHEMA

    @property
    def _engine(self):
        return RawDatasetStorage.metadata.bind


# Mixin implementations for convenience
class BasePandasRawDataset(BaseRawDataset, PandasDatasetMixin):
    pass

class BaseNumpyRawDataset(BaseRawDataset, NumpyDatasetMixin):
    pass
