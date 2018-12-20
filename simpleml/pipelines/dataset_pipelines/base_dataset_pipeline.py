from simpleml.persistables.meta_registry import DatasetPipelineRegistry
from simpleml.persistables.guid import GUID
from simpleml.pipelines.base_pipeline import BasePipeline
from simpleml.pipelines.validation_split_mixins import NoSplitMixin, ExplicitSplitMixin
from sqlalchemy import Column, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from future.utils import with_metaclass

__author__ = 'Elisha Yadgaran'


class AbstractBaseDatasetPipeline(with_metaclass(DatasetPipelineRegistry, BasePipeline)):
    '''
    Abstract Base class for all Dataset Pipeline objects.
    '''
    __abstract__ = True


class BaseDatasetPipeline(AbstractBaseDatasetPipeline):
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
class BaseNoSplitDatasetPipeline(BaseDatasetPipeline, NoSplitMixin):
    pass

class BaseExplicitSplitDatasetPipeline(BaseDatasetPipeline, ExplicitSplitMixin):
    pass
