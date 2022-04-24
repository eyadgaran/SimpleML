"""
ORM module for model objects
"""

__author__ = "Elisha Yadgaran"


import logging

from sqlalchemy import Column, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import relationship

from simpleml.orm.persistable import ORMPersistable
from simpleml.orm.pipeline import ORMPipeline
from simpleml.orm.sqlalchemy_types import GUID, MutableJSON

LOGGER = logging.getLogger(__name__)


class ORMModel(ORMPersistable):
    """
    Base class for all Model objects. Defines the required
    parameters for versioning and all other metadata can be
    stored in the arbitrary metadata field

    -------
    Schema
    -------
    params: model parameter metadata for easy insight into hyperparameters across trainings
    feature_metadata: metadata insight into resulting features and importances

    pipeline_id: foreign key relation to the pipeline used to transform input to the model
        (training is also dependent on originating dataset but scoring only needs access to the pipeline)
    """

    __tablename__ = "models"

    # Additional model specific metadata
    params = Column(MutableJSON)
    feature_metadata = Column(MutableJSON)

    # Only dependency is the pipeline (to score in production)
    pipeline_id = Column(
        GUID, ForeignKey("pipelines.id", name="models_pipeline_id_fkey")
    )
    pipeline = relationship("ORMPipeline", enable_typechecks=False)

    __table_args__ = (
        # Unique constraint for versioning
        UniqueConstraint("name", "version", name="model_name_version_unique"),
        # Index for searching through friendly names
        Index("model_name_index", "name"),
    )

    @classmethod
    def load_pipeline(cls, id: str):
        return cls.load_reference(ORMPipeline, id)
