"""
ORM module for metric objects
"""

__author__ = "Elisha Yadgaran"


import logging

from sqlalchemy import Column, ForeignKey, Index, UniqueConstraint, func
from sqlalchemy.orm import relationship

from simpleml.orm.dataset import ORMDataset
from simpleml.orm.model import ORMModel
from simpleml.orm.persistable import ORMPersistable
from simpleml.orm.sqlalchemy_types import GUID, MutableJSON

LOGGER = logging.getLogger(__name__)


class ORMMetric(ORMPersistable):
    """
    Abstract Base class for all Metric objects

    -------
    Schema
    -------
    name: the metric name
    values: JSON object with key: value pairs for performance on test dataset
        (ex: FPR: TPR to create ROC Curve)
        Singular value metrics take the form - {'agg': value}
    model_id: foreign key to the model that was used to generate predictions
    dataset_id:
    """

    __tablename__ = "metrics"

    values = Column(MutableJSON, nullable=False)

    # Dependencies are model and dataset
    model_id = Column(GUID, ForeignKey("models.id", name="metrics_model_id_fkey"))
    model = relationship("ORMModel", enable_typechecks=False)
    dataset_id = Column(GUID, ForeignKey("datasets.id", name="metrics_dataset_id_fkey"))
    dataset = relationship("ORMDataset", enable_typechecks=False)

    __table_args__ = (
        # Metrics don't have the notion of versions, values should be deterministic
        # by class, model, and dataset - name should be the combination of class and dataset
        # Still exists to stay consistent with the persistables style of unrestricted duplication
        # (otherwise would be impossible to distinguish a duplicated metric -- name and model_id would be the same)
        # Unique constraint for versioning
        UniqueConstraint(
            "name", "model_id", "version", name="metric_name_model_version_unique"
        ),
        # Index for searching through friendly names
        Index("metric_name_index", "name"),
    )

    @classmethod
    def get_latest_version(cls, name: str, model_id: str) -> int:
        """
        Versions should be autoincrementing for each object (constrained over
        friendly name and model). Executes a database lookup and increments..
        """
        last_version = (
            cls.query_by(func.max(cls.version))
            .filter(cls.name == name, cls.model_id == model_id)
            .scalar()
        )

        if last_version is None:
            last_version = 0

        return last_version + 1

    @classmethod
    def load_dataset(cls, id: str):
        return cls.load_reference(ORMDataset, id)

    @classmethod
    def load_model(cls, id: str):
        return cls.load_reference(ORMModel, id)
