__author__ = "Elisha Yadgaran"


import logging
import weakref
from typing import Any

from simpleml.datasets.base_dataset import Dataset
from simpleml.models.base_model import Model
from simpleml.persistables.base_persistable import Persistable
from simpleml.registries import MetricRegistry
from simpleml.utils.errors import MetricError

LOGGER = logging.getLogger(__name__)


class Metric(Persistable, metaclass=MetricRegistry):
    '''
    Base class for all Metric objects
    '''
    object_type: str = 'METRIC'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def add_dataset(self, dataset: Dataset) -> None:
        """
        Setter method for dataset used
        """
        self.dataset = dataset

    def add_model(self, model: Model) -> None:
        """
        Setter method for model used
        """
        self.model = model

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
        '''
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
