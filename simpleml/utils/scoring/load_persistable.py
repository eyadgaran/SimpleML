"""
Module to query registry and retrieve persistables from wherever they are
stored.
"""

__author__ = "Elisha Yadgaran"

import logging
from typing import Any, Dict

from simpleml.datasets.base_dataset import Dataset
from simpleml.metrics.base_metric import Metric
from simpleml.models.base_model import Model
from simpleml.orm.dataset import ORMDataset
from simpleml.orm.metric import ORMMetric
from simpleml.orm.model import ORMModel
from simpleml.orm.persistable import ORMPersistable
from simpleml.orm.pipeline import ORMPipeline
from simpleml.persistables.base_persistable import Persistable
from simpleml.pipelines.base_pipeline import Pipeline
from simpleml.utils.errors import SimpleMLError
from simpleml.utils.library_versions import INSTALLED_LIBRARIES

LOGGER = logging.getLogger(__name__)


class PersistableLoader(object):
    """
    Wrapper class to load various persistables

    Sqlalchemy-mixins active record style allows for keyword based filtering:
        `BaseClass.where(**filters).order_by(**ordering).first()`
    """

    @classmethod
    def load_persistable(cls, persistable_class: ORMPersistable, filters: Dict[str, Any]) -> Persistable:
        record = persistable_class.where(**filters).order_by(persistable_class.version.desc()).first()
        if record is not None:
            persistable = record.load(load_externals=False)
            cls.validate_environment(persistable)
            return persistable
        else:
            raise SimpleMLError(
                "No persistable found for specified filters: {}".format(filters)
            )

    @classmethod
    def load_dataset(cls, **filters) -> Dataset:
        return cls.load_persistable(ORMDataset, filters)

    @classmethod
    def load_pipeline(cls, **filters) -> Pipeline:
        return cls.load_persistable(ORMPipeline, filters)

    @classmethod
    def load_model(cls, **filters) -> Model:
        return cls.load_persistable(ORMModel, filters)

    @classmethod
    def load_metric(cls, **filters) -> Metric:
        return cls.load_persistable(ORMMetric, filters)

    @staticmethod
    def validate_environment(persistable: Persistable) -> None:
        training_env = persistable.library_versions
        scoring_env = INSTALLED_LIBRARIES

        mismatches = []

        for package, version in training_env.items():
            if package not in scoring_env:
                mismatches.append(
                    {"package": package, "expected_version": version, "version": "None"}
                )
            else:
                if version != scoring_env[package]:
                    mismatches.append(
                        {
                            "package": package,
                            "expected_version": version,
                            "version": scoring_env[package],
                        }
                    )

        warning_msg = "Attempted to score with different dependencies than training, proceed at your own risk"

        if mismatches:
            LOGGER.warning(warning_msg)
            for mismatch in mismatches:
                LOGGER.warning(
                    "Expected: {package}=={expected_version}, found: {version}".format(
                        **mismatch
                    )
                )
