"""
ORM module for dataset objects
"""

__author__ = "Elisha Yadgaran"


import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import relationship

from simpleml.orm.persistable import ORMPersistable
from simpleml.orm.sqlalchemy_types import GUID

LOGGER = logging.getLogger(__name__)


class ORMDataset(ORMPersistable):
    """
    Base class for all  Dataset objects.

    -------
    Schema
    -------
    pipeline_id: foreign key relation to the dataset pipeline used as input
    """

    __tablename__ = "datasets"

    pipeline_id = Column(GUID, ForeignKey("pipelines.id"))
    pipeline = relationship(
        "ORMPipeline", enable_typechecks=False, foreign_keys=[pipeline_id]
    )

    __table_args__ = (
        # Unique constraint for versioning
        UniqueConstraint("name", "version", name="dataset_name_version_unique"),
        # Index for searching through friendly names
        Index("dataset_name_index", "name"),
    )

    @classmethod
    def load_pipeline(cls, id: str):
        # avoid circular import
        from simpleml.orm.pipeline import ORMPipeline

        return cls.load_reference(ORMPipeline, id)
