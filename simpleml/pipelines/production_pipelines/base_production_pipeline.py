from simpleml.persistables.meta_registry import PipelineRegistry
from simpleml.persistables.guid import GUID
from simpleml.pipelines.base_pipeline import BasePipeline
from simpleml.pipelines.validation_split_mixins import NoSplitMixin, RandomSplitMixin,\
    ChronologicalSplitMixin, ExplicitSplitMixin
from sqlalchemy import Column, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from future.utils import with_metaclass

__author__ = 'Elisha Yadgaran'


class AbstractBaseProductionPipeline(with_metaclass(PipelineRegistry, BasePipeline)):
    '''
    Abstract Base class for all Production Pipeline objects.
    '''
    __abstract__ = True


class BaseProductionPipeline(AbstractBaseProductionPipeline):
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


# Mixin implementations for convenience
class BaseNoSplitProductionPipeline(BaseProductionPipeline, NoSplitMixin):
    pass

class BaseExplicitSplitProductionPipeline(BaseProductionPipeline, ExplicitSplitMixin):
    pass

class BaseRandomSplitProductionPipeline(RandomSplitMixin, BaseProductionPipeline):
    # Needs to be used as base class because of MRO initialization
    pass

class BaseChronologicalSplitProductionPipeline(ChronologicalSplitMixin, BaseProductionPipeline):
    # Needs to be used as base class because of MRO initialization
    pass
