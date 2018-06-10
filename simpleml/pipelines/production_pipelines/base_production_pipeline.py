from simpleml.persistables.guid import GUID
from simpleml.pipelines.base_pipeline import BasePipeline
from sqlalchemy import Column, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship

__author__ = 'Elisha Yadgaran'


class BaseProductionPipeline(BasePipeline):
    '''
    Base class for all Production Pipeline objects.

    -------
    Schema
    -------
    dataset_id: foreign key relation to the dataset used as input
    '''
    __tablename__ = 'pipelines'

    dataset_id = Column(GUID, ForeignKey("datasets.id"))
    dataset = relationship("BaseProcessedDataset", enable_typechecks=False)

    __table_args__ = (
        # Unique constraint for versioning
        UniqueConstraint('name', 'version', name='pipeline_name_version_unique'),
        # Index for searching through friendly names
        Index('pipeline_name_index', 'name'),
     )
