"""
Util to recalculate persistable hashes
"""

__author__ = "Elisha Yadgaran"

import logging
from queue import SimpleQueue
from typing import List, Optional

from simpleml.datasets.base_dataset import Dataset
from simpleml.metrics.base_metric import Metric
from simpleml.models.base_model import Model
from simpleml.persistables.base_persistable import Persistable
from simpleml.persistables.hashing import CustomHasherMixin
from simpleml.pipelines.base_pipeline import Pipeline
from simpleml.registries import SIMPLEML_REGISTRY

LOGGER = logging.getLogger(__name__)


class HashRecalculator(object):
    """
    Utility class to recalculate hashes for persistables

    Useful for backfilling changes to hash logic and for database migrations
    that impact fields included in the hash (e.g. config metadata)

    Expects to be called as part of an active session

    ```
    HashRecalculator(
        fail_on_error=False,
        recursively_recalculate_dependent_hashes=True
    ).run()
    ```
    """

    def __init__(
        self,
        fail_on_error: bool,
        recursively_recalculate_dependent_hashes: bool,
        dataset_ids: Optional[List[str]] = None,
        pipeline_ids: Optional[List[str]] = None,
        model_ids: Optional[List[str]] = None,
        metric_ids: Optional[List[str]] = None,
    ):
        self.fail_on_error = fail_on_error
        self.recursively_recalculate_dependent_hashes = (
            recursively_recalculate_dependent_hashes
        )

        # persistable queues
        self.dataset_queue = self.ids_to_records(Dataset, dataset_ids)
        self.pipeline_queue = self.ids_to_records(Pipeline, pipeline_ids)
        self.model_queue = self.ids_to_records(Model, model_ids)
        self.metric_queue = self.ids_to_records(Metric, metric_ids)

    def ids_to_records(
        self, persistable_cls: Persistable, ids: Optional[List[str]]
    ) -> List[Persistable]:
        records = SimpleQueue()
        if ids is not None:
            for id in ids:
                records.put(persistable_cls.find(id))
        return records

    def run(self) -> None:
        _iterations = 1
        session = Persistable._session
        with session.begin():  # automatic rollback if error raised
            while not self.is_finished:
                LOGGER.debug(f"Processing iteration {_iterations}")
                _iterations += 1
                self.process_queue(self.dataset_queue)
                self.process_queue(self.pipeline_queue)
                self.process_queue(self.model_queue)
                self.process_queue(self.metric_queue)

    @property
    def is_finished(self):
        return (
            self.dataset_queue.empty()
            and self.pipeline_queue.empty()
            and self.model_queue.empty()
            and self.metric_queue.empty()
        )

    def process_queue(self, queue: SimpleQueue) -> None:
        """
        Loop one iteration through a queue -- adds items back to queues if
        recursive parameter set
        """
        LOGGER.debug(f"Processing {queue.qsize()} items in queue")
        while not queue.empty():
            record = queue.get()
            existing_hash = record.hash_
            new_hash = self.recalculate_hash(record)
            if existing_hash == new_hash:
                LOGGER.debug("No hash modification, skipping update")
                continue
            LOGGER.debug(
                f"Updating persistable {record.id} hash {existing_hash} -> {new_hash}"
            )
            record.update(hash_=new_hash)

            if self.recursively_recalculate_dependent_hashes:
                self.queue_dependent_persistables(record)

    def recalculate_hash(self, record):
        try:
            # turn record into a persistable with a hash method
            record.load(load_externals=False)
            return record._hash()
        except Exception as e:
            if self.fail_on_error:
                raise
            else:
                LOGGER.error(
                    f"Failed to generate a new hash for record, skipping modification; {e}"
                )
                return record.hash_

    def queue_dependent_persistables(self, persistable: Persistable) -> None:
        """
        Queries for dependent persistables and queues them into the respective
        queues
        """
        persistable_type = persistable.object_type

        # downstream dependencies
        dependency_map = {
            "DATASET": (
                (Pipeline, "dataset_id", self.pipeline_queue),
                (Metric, "dataset_id", self.metric_queue),
            ),
            "PIPELINE": (
                (Dataset, "pipeline_id", self.dataset_queue),
                (Model, "pipeline_id", self.model_queue),
            ),
            "MODEL": ((Metric, "model_id", self.metric_queue),),
            "METRIC": (),
        }

        for (immediate_dependency, foreign_key, queue) in dependency_map[
            persistable_type
        ]:
            dependents = immediate_dependency.where(
                **{foreign_key: persistable.id}
            ).all()
            LOGGER.debug(
                f"Found {len(dependents)} dependent persistables. Adding to queues"
            )
            for dependent in dependents:
                queue.put(dependent)


def recalculate_dataset_hashes(
    fail_on_error: bool = False, recursively_recalculate_dependent_hashes: bool = False
) -> None:
    """
    Convenience helper to recompute dataset hashes. Optionally recalculates hashes
    for downstream persistables
    """
    records = Dataset.all()
    recalculator = HashRecalculator(
        fail_on_error=fail_on_error,
        recursively_recalculate_dependent_hashes=recursively_recalculate_dependent_hashes,
        dataset_ids=[i.id for i in records],
    )
    recalculator.run()


def recalculate_pipeline_hashes(
    fail_on_error: bool = False, recursively_recalculate_dependent_hashes: bool = False
) -> None:
    """
    Convenience helper to recompute pipeline hashes. Optionally recalculates hashes
    for downstream persistables
    """
    records = Pipeline.all()
    recalculator = HashRecalculator(
        fail_on_error=fail_on_error,
        recursively_recalculate_dependent_hashes=recursively_recalculate_dependent_hashes,
        dataset_ids=[i.id for i in records],
    )
    recalculator.run()


def recalculate_model_hashes(
    fail_on_error: bool = False, recursively_recalculate_dependent_hashes: bool = False
) -> None:
    """
    Convenience helper to recompute model hashes. Optionally recalculates hashes
    for downstream persistables
    """
    records = Model.all()
    recalculator = HashRecalculator(
        fail_on_error=fail_on_error,
        recursively_recalculate_dependent_hashes=recursively_recalculate_dependent_hashes,
        dataset_ids=[i.id for i in records],
    )
    recalculator.run()


def recalculate_metric_hashes(
    fail_on_error: bool = False, recursively_recalculate_dependent_hashes: bool = False
) -> None:
    """
    Convenience helper to recompute metric hashes. Optionally recalculates hashes
    for downstream persistables
    """
    records = Metric.all()
    recalculator = HashRecalculator(
        fail_on_error=fail_on_error,
        recursively_recalculate_dependent_hashes=recursively_recalculate_dependent_hashes,
        dataset_ids=[i.id for i in records],
    )
    recalculator.run()
