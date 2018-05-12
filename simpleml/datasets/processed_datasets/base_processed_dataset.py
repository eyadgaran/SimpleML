from simpleml.datasets.base_dataset import BaseDataset
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
