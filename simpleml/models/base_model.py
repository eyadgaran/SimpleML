from simpleml.persistables.base_persistable import BasePersistable
from sqlalchemy import Column, String, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

__author__ = 'Elisha Yadgaran'


class BaseModel(BasePersistable):
    '''
    Base class for all Model objects. Defines the required
    parameters for versioning and all other metadata can be
    stored in the arbitrary metadata field

    -------
    Schema
    -------
    author: model creator
    section: organizational attribute to manage many models pertaining to a single grouping
        ex: partitioning on an attribute and training an individual model for
        each instance (instead of one model with the attribute as a feature)
    name: the name of this model - primary way of tracking evolution of performance
        across versions
    version: string version 'x.y.z' of the model
    version_description: description that explains what is new or different about this version
    pipeline_id: foreign key relation to the pipeline used to transform input to the model
        (training is also dependent on originating dataset but scoring only needs access to the pipeline)
    params: model parameter metadata for easy insight into hyperparameters across trainings
    feature_metadata: metadata insight into resulting features and importances
    '''
    __tablename__ = 'models'

    author = Column(String, default='default')
    section = Column(String, default='default')
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    version_description = Column(String, default='')

    # Only dependency is the pipeline (to score in production)
    pipeline_id = Column(String, ForeignKey("pipelines.id"))
    pipeline = relationship("BasePipeline")

    # Additional model specific metadata
    params = Column(JSONB, default={})
    feature_metadata = Column(JSONB, default={})
