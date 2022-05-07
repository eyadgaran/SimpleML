"""
ORM module for pipeline objects
"""

__author__ = "Elisha Yadgaran"

import logging

from sqlalchemy import Column, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import relationship

from simpleml.orm.dataset import ORMDataset
from simpleml.orm.persistable import ORMPersistable
from simpleml.orm.sqlalchemy_types import GUID, MutableJSON

LOGGER = logging.getLogger(__name__)


class ORMPipeline(ORMPersistable):
    """
    Base class for all Pipeline objects.

    -------
    Schema
    -------
    params: pipeline parameter metadata for easy insight into hyperparameters across trainings
    dataset_id: foreign key relation to the dataset used as input
    """

    __tablename__ = "pipelines"

    # Additional pipeline specific metadata
    params = Column(MutableJSON)

    dataset_id = Column(
        GUID, ForeignKey("datasets.id", name="pipelines_dataset_id_fkey")
    )
    dataset = relationship(
        "ORMDataset", enable_typechecks=False, foreign_keys=[dataset_id]
    )

    __table_args__ = (
        # Unique constraint for versioning
        UniqueConstraint("name", "version", name="pipeline_name_version_unique"),
        # Index for searching through friendly names
        Index("pipeline_name_index", "name"),
    )

    @classmethod
    def load_dataset(cls, id: str):
        return cls.load_reference(ORMDataset, id)
