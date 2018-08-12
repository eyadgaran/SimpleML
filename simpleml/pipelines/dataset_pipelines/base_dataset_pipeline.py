from simpleml.persistables.guid import GUID
from simpleml.pipelines.base_pipeline import BasePipeline
from simpleml.pipelines.validation_split_mixins import NoSplitMixin
from sqlalchemy import Column, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship

__author__ = 'Elisha Yadgaran'


class BaseDatasetPipeline(BasePipeline):
    '''
    Base class for all Dataset Pipeline objects.

    -------
    Schema
    -------
    dataset_id: foreign key relation to the dataset used as input
    '''
    __tablename__ = 'dataset_pipelines'

    dataset_id = Column(GUID, ForeignKey("raw_datasets.id"))
    dataset = relationship("BaseRawDataset", enable_typechecks=False)

    __table_args__ = (
        # Unique constraint for versioning
        UniqueConstraint('name', 'version', name='dataset_pipeline_name_version_unique'),
        # Index for searching through friendly names
        Index('dataset_pipeline_name_index', 'name'),
     )


# Mixin implementations for convenience
# Needs to be used as base class because of MRO initialization
class BaseNoSplitDatasetPipeline(NoSplitMixin, BaseDatasetPipeline):
    pass
