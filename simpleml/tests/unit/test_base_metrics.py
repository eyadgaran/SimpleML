"""
Base metric tests
"""

__author__ = "Elisha Yadgaran"


import unittest
from unittest.mock import MagicMock

from simpleml.metrics.base_metric import Metric


class BaseMetricTests(unittest.TestCase):
    def test_default_hash_logic_with_precomputed_hash(self):
        model_mock = MagicMock()
        model_mock.hash_ = "mock_model_hash"

        dataset_mock = MagicMock()
        dataset_mock.hash_ = "mock_dataset_hash"

        metric = Metric()
        metric.add_model(model_mock)
        metric.add_dataset(dataset_mock)

        expected_hash = "a2f51ca6e1984d33e897a87b14b30ebf"
        self.assertEqual(metric._hash(), expected_hash)

    def test_default_hash_logic_with_new_hash(self):
        model_mock = MagicMock()
        model_mock.hash_ = None
        model_mock._hash.return_value = "mock_model_hash"

        dataset_mock = MagicMock()
        dataset_mock.hash_ = None
        dataset_mock._hash.return_value = "mock_dataset_hash"

        metric = Metric()
        metric.add_model(model_mock)
        metric.add_dataset(dataset_mock)

        expected_hash = "a2f51ca6e1984d33e897a87b14b30ebf"
        self.assertEqual(metric._hash(), expected_hash)

    def test_getting_pipeline_split(self):
        model_mock = MagicMock()
        pipeline_mock = MagicMock()
        dataset_split_mock = MagicMock()
        model_mock.pipeline = pipeline_mock
        pipeline_mock.get_dataset_split.return_value = dataset_split_mock
        dataset_split_mock.a = "abc"

        metric = Metric()
        metric.add_model(model_mock)

        self.assertEqual(
            metric._get_pipeline_split(column="a", split="b", other="other"), "abc"
        )
        pipeline_mock.get_dataset_split.assert_called_once_with(
            split="b", other="other"
        )

    def test_getting_dataset_split(self):
        dataset_mock = MagicMock()
        dataset_mock.get.return_value = "abc"

        metric = Metric()
        metric.add_dataset(dataset_mock)

        self.assertEqual(
            metric._get_dataset_split(column="a", split="b", other="other"), "abc"
        )
        dataset_mock.get.assert_called_once_with(column="a", split="b", other="other")


if __name__ == "__main__":
    unittest.main(verbosity=2)
