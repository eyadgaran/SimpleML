'''
Classification metric tests
'''

__author__ = 'Elisha Yadgaran'


import unittest
import pandas as pd
import numpy as np

from unittest.mock import MagicMock, patch

from simpleml.utils.errors import MetricError
from simpleml.constants import TRAIN_SPLIT, VALIDATION_SPLIT
from simpleml.metrics.classification import ClassificationMetric, BinaryClassificationMetric


class BaseClassificationTests(unittest.TestCase):
    def test_naming(self):
        '''
        Train and validation splits get renamed
        '''
        with self.subTest(dataset_split=TRAIN_SPLIT):
            metric = ClassificationMetric(dataset_split=TRAIN_SPLIT)
            self.assertEqual(metric.name, 'in_sample_')
        with self.subTest(dataset_split=VALIDATION_SPLIT):
            metric = ClassificationMetric(dataset_split=VALIDATION_SPLIT)
            self.assertEqual(metric.name, 'validation_')

    def test_getting_split_if_using_training_dataset(self):
        '''
        Expects to route via pipeline split to preserve train/test splits
        '''
        model_mock = MagicMock()
        pipeline_mock = MagicMock()
        pipeline_mock.dataset_id = '123'
        dataset_split_mock = MagicMock()
        model_mock.pipeline = pipeline_mock
        pipeline_mock.get_dataset_split.return_value = dataset_split_mock
        dataset_split_mock.a = 'abc'
        dataset_mock = MagicMock()
        dataset_mock.id = '123'
        dataset_mock.get.return_value = 'def'

        metric = ClassificationMetric()
        metric.add_model(model_mock)
        metric.add_dataset(dataset_mock)

        self.assertEqual(metric._get_split('a'), 'abc')
        pipeline_mock.get_dataset_split.assert_called_once_with(split=None)

    def test_getting_split_if_not_using_training_dataset(self):
        '''
        Expects to blindly use the dataset data
        '''
        model_mock = MagicMock()
        pipeline_mock = MagicMock()
        pipeline_mock.dataset_id = '123'
        dataset_split_mock = MagicMock()
        model_mock.pipeline = pipeline_mock
        pipeline_mock.get_dataset_split.return_value = dataset_split_mock
        dataset_split_mock.a = 'abc'
        dataset_mock = MagicMock()
        dataset_mock.id = '456'
        dataset_mock.get.return_value = 'def'

        metric = ClassificationMetric()
        metric.add_model(model_mock)
        metric.add_dataset(dataset_mock)

        self.assertEqual(metric._get_split('a'), 'def')
        self.assertFalse(pipeline_mock.get_dataset_split.called)
        dataset_mock.get.assert_called_once_with(column='a', split=None)

    def test_getting_labels(self):
        model_mock = MagicMock()
        pipeline_mock = MagicMock()
        pipeline_mock.dataset_id = '123'
        dataset_split_mock = MagicMock()
        model_mock.pipeline = pipeline_mock
        pipeline_mock.get_dataset_split.return_value = dataset_split_mock
        dataset_split_mock.a = 'abc'
        dataset_mock = MagicMock()
        dataset_mock.id = '456'
        dataset_mock.get.return_value = 'def'

        metric = ClassificationMetric()
        metric.add_model(model_mock)
        metric.add_dataset(dataset_mock)

        self.assertEqual(metric._get_split('y'), 'def')
        self.assertFalse(pipeline_mock.get_dataset_split.called)
        dataset_mock.get.assert_called_once_with(column='y', split=None)

    def test_label_error_without_dataset(self):
        metric = ClassificationMetric()
        with self.assertRaises(MetricError):
            metric.labels

    def test_probability_error_without_dataset(self):
        metric = ClassificationMetric()
        with self.assertRaises(MetricError):
            metric.probabilities

    def test_prediction_error_without_dataset(self):
        metric = ClassificationMetric()
        with self.assertRaises(MetricError):
            metric.predictions

    @patch.object(ClassificationMetric, '_get_split')
    def test_getting_invalid_probabilities(self, mock_split):
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = None
        metric = ClassificationMetric()
        metric.add_dataset('abc')
        metric.add_model(mock_model)

        mock_split.return_value = '123'
        with self.assertRaises(MetricError):
            metric.probabilities
        mock_model.predict_proba.assert_called_once_with(X='123', transform=True)

    @patch.object(ClassificationMetric, '_get_split')
    def test_getting_valid_probabilities(self, mock_split):
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[1, 2], [3, 4]])
        metric = ClassificationMetric()
        metric.add_dataset('abc')
        metric.add_model(mock_model)

        mock_split.return_value = '123'
        np.testing.assert_array_equal(metric.probabilities, np.array([[1, 2], [3, 4]]))
        mock_model.predict_proba.assert_called_once_with(X='123', transform=True)

    @patch.object(ClassificationMetric, '_get_split')
    def test_getting_invalid_predictions(self, mock_split):
        mock_model = MagicMock()
        mock_model.predict.return_value = None
        metric = ClassificationMetric()
        metric.add_dataset('abc')
        metric.add_model(mock_model)

        mock_split.return_value = '123'
        with self.assertRaises(MetricError):
            metric.predictions
        mock_model.predict.assert_called_once_with(X='123', transform=True)

    @patch.object(ClassificationMetric, '_get_split')
    def test_getting_valid_predictions(self, mock_split):
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[1, 2], [3, 4]])
        metric = ClassificationMetric()
        metric.add_dataset('abc')
        metric.add_model(mock_model)

        mock_split.return_value = '123'
        np.testing.assert_array_equal(metric.predictions, np.array([[1, 2], [3, 4]]))
        mock_model.predict.assert_called_once_with(X='123', transform=True)

    def test_prediction_validation_logic(self):
        for bad_type in [None, pd.DataFrame(), pd.Series(), np.array([]), np.ndarray([0, 0])]:
            with self.subTest(bad_type=bad_type):
                with self.assertRaises(MetricError):
                    ClassificationMetric.validate_predictions(bad_type)


class BinaryClassificationTests(unittest.TestCase):
    def test_getting_labels(self):
        pass

    def test_empty_label_validation_logic(self):
        for bad_type in [None, pd.DataFrame(), pd.Series(), np.array([]), np.ndarray([0, 0])]:
            with self.subTest(bad_type=bad_type):
                with self.assertRaises(MetricError):
                    BinaryClassificationMetric.validate_labels(bad_type)

    def test_nonbinary_label_validation_logic(self):
        for bad_type in [
            [1, 2],
            [0, 1, 2],
            pd.DataFrame([[0], [1], [2]]),
            pd.DataFrame([[0, 0], [0, 2], [1, 1]]),  # multi column handling
            pd.Series([0, 1, 2]),
            np.array([[0], [1], [2]]),
            np.array([[0, 0], [0, 2], [1, 1]]),  # multi column handling
            np.array([0, 1, 2])
        ]:
            with self.subTest(bad_type=bad_type):
                with self.assertRaises(MetricError):
                    BinaryClassificationMetric.validate_labels(bad_type)

    def test_getting_multi_column_probabilities(self):
        '''
        Should cast to single series
        '''
        pass

    def test_getting_multi_column_predictions(self):
        '''
        Should cast to single series
        '''
        pass

    def test_confusion_matrix_logic(self):
        pass

    def test_curve_deduplication(self):
        pass

    def test_true_positive_rate_calculation(self):
        pass

    def test_false_positive_rate_calculation(self):
        pass

    def test_true_negative_rate_calculation(self):
        pass

    def test_false_negative_rate_calculation(self):
        pass

    def test_false_discovery_rate_calculation(self):
        pass

    def test_false_omission_rate_calculation(self):
        pass

    def test_positive_predictive_value_calculation(self):
        pass

    def test_negative_predictive_value_calculation(self):
        pass

    def test_predicted_positive_rate_calculation(self):
        pass

    def test_predicted_negative_rate_calculation(self):
        pass

    def test_accuracy_calculation(self):
        pass

    def test_f1_calculation(self):
        pass

    def test_matthews_correlation_coefficient_calculation(self):
        pass

    def test_informedness_calculation(self):
        pass

    def test_markedness_calculation(self):
        pass


if __name__ == '__main__':
    unittest.main(verbosity=2)
