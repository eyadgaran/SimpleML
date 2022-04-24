__author__ = "Elisha Yadgaran"


import logging
import uuid
import weakref
from typing import Any, Optional, Union

from simpleml.datasets.base_dataset import Dataset
from simpleml.models.base_model import Model
from simpleml.persistables.base_persistable import Persistable
from simpleml.registries import MetricRegistry
from simpleml.utils.errors import MetricError

LOGGER = logging.getLogger(__name__)


class Metric(Persistable, metaclass=MetricRegistry):
    """
    Base class for all Metric objects
    """

    object_type: str = "METRIC"

    def __init__(
        self,
        dataset_id: Optional[Union[str, uuid.uuid4]] = None,
        model_id: Optional[Union[str, uuid.uuid4]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # initialize null references
        self.dataset_id = dataset_id
        self.model_id = model_id

    def add_dataset(self, dataset: Dataset) -> None:
        """
        Setter method for dataset used
        """
        if dataset is None:
            return
        self.dataset_id = dataset.id
        self.dataset = dataset

    @property
    def dataset(self):
        """
        Use a weakref to bind linked dataset so it doesnt bloat usage
        returns dataset if still available or tries to fetch otherwise
        """
        # still referenced weakref
        if hasattr(self, "_dataset") and self._dataset() is not None:
            return self._dataset()

        # null return if no associated dataset (governed by dataset_id)
        elif self.dataset_id is None:
            return None

        # else regenerate weakref
        LOGGER.info("No referenced object available. Refreshing weakref")
        dataset = self._load_dataset()
        self._dataset = weakref.ref(dataset)
        return dataset

    @dataset.setter
    def dataset(self, dataset: Dataset) -> None:
        """
        Need to be careful not to set as the orm object
        Cannot load if wrong type because of recursive behavior (will
        propagate down the whole dependency chain)
        """
        self._dataset = weakref.ref(dataset)

    def _load_dataset(self):
        """
        Helper to fetch the dataset
        """
        return self.orm_cls.load_dataset(self.dataset_id)

    def add_model(self, model: Model) -> None:
        """
        Setter method for model used
        """
        if model is None:
            return
        self.model_id = model.id
        self.model = model

    @property
    def model(self):
        """
        Use a weakref to bind linked model so it doesnt bloat usage
        returns model if still available or tries to fetch otherwise
        """
        # still referenced weakref
        if hasattr(self, "_model") and self._model() is not None:
            return self._model()

        # null return if no associated model (governed by model_id)
        elif self.model_id is None:
            return None

        # else regenerate weakref
        LOGGER.info("No referenced object available. Refreshing weakref")
        model = self._load_model()
        self._model = weakref.ref(model)
        return model

    @model.setter
    def model(self, model: Model) -> None:
        """
        Need to be careful not to set as the orm object
        Cannot load if wrong type because of recursive behavior (will
        propagate down the whole dependency chain)
        """
        self._model = weakref.ref(model)

    def _load_model(self):
        """
        Helper to fetch the model
        """
        return self.orm_cls.load_model(self.model_id)

    def _hash(self) -> str:
        """
        Hash is the combination of the:
            1) Model
            2) Dataset (optional)
            3) Metric
            4) Config
        """
        model_hash = self.model.hash_ or self.model._hash()
        if self.dataset is not None:
            dataset_hash = self.dataset.hash_ or self.dataset._hash()
        else:
            dataset_hash = None
        metric = self.__class__.__name__
        config = self.config

        return self.custom_hasher((model_hash, dataset_hash, metric, config))

    def _get_latest_version(self) -> int:
        """
        Versions should be autoincrementing for each object (constrained over
        friendly name and model). Executes a database lookup and increments..
        """
        return self.orm_cls.get_latest_version(name=self.name, model_id=self.model.id)

    def _get_pipeline_split(self, column: str, split: str, **kwargs) -> Any:
        """
        For special case where dataset is the same as the model's dataset, the
        dataset splits can refer to the pipeline imposed splits, not the inherent
        dataset's splits. Use the pipeline split then
        ex: RandomSplitPipeline on NoSplitDataset evaluating "in_sample" performance
        """
        return getattr(
            self.model.pipeline.get_dataset_split(split=split, **kwargs), column
        )

    def _get_dataset_split(self, **kwargs) -> Any:
        """
        Default accessor for dataset data. REFERS TO RAW DATASETS
        not the pipelines superimposed. That means that datasets that do not
        define explicit splits will have no notion of downstream splits
        (e.g. RandomSplitPipeline)
        """
        return self.dataset.get(**kwargs)

    def save(self, **kwargs) -> None:
        """
        Extend parent function with a few additional save routines
        """
        if self.model is None:
            raise MetricError("Must set model before saving")

        if self.values is None:
            raise MetricError("Must score metric before saving")

        super().save(**kwargs)

    def score(self, **kwargs):
        """
        Abstract method for each metric to define

        Should set self.values
        """
        raise NotImplementedError
