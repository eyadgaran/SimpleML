from simpleml.datasets.base_dataset import BaseDataset
from simpleml.persistables.dataset_storage import RawDatasetStorage, RAW_DATASET_SCHEMA
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

    __table_args__ = (
        # Unique constraint for versioning
        UniqueConstraint('name', 'version', name='raw_dataset_name_version_unique'),
        # Index for searching through friendly names
        Index('raw_dataset_name_index', 'name'),
     )

    def _save_external_files(self):
        super(BaseRawDataset, self)._save_external_files(
            RAW_DATASET_SCHEMA, RawDatasetStorage.metadata.bind)

    def _load_external_files(self):
        super(BaseRawDataset, self)._load_external_files(RawDatasetStorage.metadata.bind)


class QueryableRawDataset(BaseRawDataset):
    '''subclass to implement dummy abstract methods'''
    def build_dataframe(self):
        raise "Dont use me"
