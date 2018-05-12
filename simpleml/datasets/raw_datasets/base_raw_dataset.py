from simpleml.datasets.base_dataset import BaseDataset
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
