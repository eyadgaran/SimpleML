from simpleml.datasets.base_dataset import BaseDataset
from simpleml.persistables.dataset_storage import DatasetStorage, DATASET_SCHEMA
from sqlalchemy import UniqueConstraint, Index

__author__ = 'Elisha Yadgaran'


class BaseProcessedDataset(BaseDataset):
    '''
    Base class for all Processed Dataset objects.

    -------
    Schema
    -------
    No additional columns
    '''

    __tablename__ = 'datasets'

    __table_args__ = (
        # Unique constraint for versioning
        UniqueConstraint('name', 'version', name='dataset_name_version_unique'),
        # Index for searching through friendly names
        Index('dataset_name_index', 'name'),
     )

    def _save_external_files(self):
        super(BaseProcessedDataset, self)._save_external_files(
            DATASET_SCHEMA, DatasetStorage.metadata.bind)

    def _load_external_files(self):
        super(BaseProcessedDataset, self)._load_external_files(DatasetStorage.metadata.bind)
