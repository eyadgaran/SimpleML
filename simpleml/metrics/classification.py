'''
Module for classification metrics
https://en.wikipedia.org/wiki/Confusion_matrix

Includes base class and derived metrics following the nomenclature:
    ConstraintValueMetric
Where:
    Constraint is the lookup criteria (ex FPR in ROC curve)
    Value is desired value (ex TPR in ROC curve)


This module is organized by metric and prediction dependencies:
    1) Base classes with methods and utilities
    2) Aggregate Metrics (single value output)
        2a) Single values computed via Predict method (operating points)
        2b) Single values computed via proba method (agg over curve)
    3) Curve Metrics (constraint: value)
        3a) Threshold: confusion matrix metrics
        3b) confusion matrix metrics: threshold or other metrics
'''

from simpleml.metrics.base_metric import Metric
from simpleml.constants import TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT
from simpleml.utils.errors import MetricError
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score
from abc import abstractmethod
import numpy as np
import pandas as pd
import logging


__author__ = 'Elisha Yadgaran'


LOGGER = logging.getLogger(__name__)


############################### BASE ###############################

class ClassificationMetric(Metric):
    '''
    TODO: Figure out multiclass generalizations
    '''

    def __init__(self, dataset_split=None, **kwargs):
        '''
        :param dataset_split: string denoting which dataset split to use
            can be one of: `TRAIN`, `VALIDATION`, Other. Other gets no prefix
            Default is train split to stay consistent with no split mapping to Train
            in Pipeline

        '''
        name = kwargs.pop('name', '')

        # Explicitly call out in sample or validation metrics
        # Only relevant if using a split dataset. No split pipelines will return
        # all data by default on null input, while split ones will return empty splits
        if dataset_split == TRAIN_SPLIT:
            name = 'in_sample_' + name
        elif dataset_split == VALIDATION_SPLIT:
            name = 'validation_' + name

        super(ClassificationMetric, self).__init__(name=name, **kwargs)
        self.config['dataset_split'] = dataset_split

    def _get_split(self, column):
        if self.dataset.id == self.model.pipeline.dataset_id:
            LOGGER.debug('Dataset is the same as model dataset, using pipeline dataset split instead of raw dataset one')
            return self._get_pipeline_split(column=column, split=self.config.get('dataset_split'))
        return self._get_dataset_split(column=column, split=self.config.get('dataset_split'))

    @property
    def labels(self):
        if self.dataset is None:
            raise MetricError('Must set dataset before scoring classification metrics!')
        return self._get_split(column='y')

    @property
    def probabilities(self):
        if self.dataset is None:
            raise MetricError('Must set dataset before scoring classification metrics!')
        probabilities = self.model.predict_proba(
            X=self._get_split(column='X'),
            transform=True
        )
        self.validate_predictions(probabilities)
        return probabilities

    @property
    def predictions(self):
        if self.dataset is None:
            raise MetricError('Must set dataset before scoring classification metrics!')
        preds = self.model.predict(
            X=self._get_split(column='X'),
            transform=True
        )
        self.validate_predictions(preds)
        return preds

    @staticmethod
    def validate_predictions(predictions):
        invalid = None
        if predictions is None:
            invalid = True
        elif isinstance(predictions, (pd.DataFrame, pd.Series)) and predictions.empty:
            invalid = True
        elif isinstance(predictions, np.ndarray) and predictions.size == 0:
            invalid = True

        if invalid:
            raise MetricError('Attempting to score an empty dataset')


class BinaryClassificationMetric(ClassificationMetric):
    @property
    def labels(self):
        # extends parent label retrieval with a validation step for binary values
        labels = super(BinaryClassificationMetric, self).labels
        self.validate_labels(labels)
        return labels

    @staticmethod
    def validate_labels(labels):
        invalid = None
        if labels is None:
            invalid = True
        else:
            invalid = (len(set(labels) - {0, 1}) > 0)

        if invalid:
            raise MetricError('Attempting to score a binary metric with labels outside of {0,1}')

    @property
    def probabilities(self):
        probabilities = super(BinaryClassificationMetric, self).probabilities
        if len(probabilities.shape) > 1 and probabilities.shape[1] > 1:
            # Indicates multiple class probabilities are returned (class_0, class_1)
            probabilities = probabilities[:, 1]
        return probabilities

    @property
    def predictions(self):
        predictions = super(BinaryClassificationMetric, self).predictions
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Indicates multiple class predictions are returned (class_0, class_1)
            predictions = predictions[:, 1]
        return predictions

    @property
    def confusion_matrix(self):
        '''
        Property method to return (or generate) dataframe of confusion
        matrix at each threshold
        '''
        if not hasattr(self, '_confusion_matrix') or self._confusion_matrix is None:
            self.create_confusion_matrix()

        return self._confusion_matrix

    @staticmethod
    def _create_confusion_matrix(thresholds, probabilities, labels):
        '''
        Independent computation method (easier testing)
        '''
        results = []
        for threshold in thresholds:
            predictions = np.where(probabilities >= threshold, 1, 0)
            tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
            results.append((threshold, tn, fp, fn, tp))

        return pd.DataFrame(results, columns=['threshold', 'tn', 'fp', 'fn', 'tp'])

    def create_confusion_matrix(self):
        '''
        Iterate through each threshold and compute confusion matrix
        '''
        # Thresholds to compute confusion matrix at (default every 0.005 increment)
        thresholds = np.linspace(0, 1, 201)
        probabilities = self.probabilities
        labels = self.labels

        self._confusion_matrix = self._create_confusion_matrix(thresholds, probabilities, labels)

    @staticmethod
    def dedupe_curve(keys, values, maximize=True, round_places=3):
        '''
        Method to deduplicate multiple values for the same key on a curve
        (ex multiple thresholds with the same fpr and different tpr for roc)

        :param maximize: Boolean, whether to choose the maximum value for each
            unique key or the minimum
        '''
        # Round arbitrary decimal places to dedupe
        keys = [round(i, round_places) for i in keys]
        values = [round(i, round_places) for i in values]

        df = pd.DataFrame(list(zip(keys, values)), columns=['keys', 'values'])
        df.dropna(axis=0, inplace=True)

        agg = 'max' if maximize else 'min'
        return df.groupby('keys').agg({'values': agg}).to_dict()['values']

    @property
    def thresholds(self):
        '''
        Convenience property for the probability thresholds
        '''
        return self.confusion_matrix.threshold

    @property
    def true_positive_rate(self):
        '''
        Convenience property for the True Positive Rate (TP/TP+FN)
        '''
        return self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)

    @property
    def false_positive_rate(self):
        '''
        Convenience property for the False Positive Rate (FP/FP+TN)
        '''
        return self.confusion_matrix.fp / (self.confusion_matrix.fp + self.confusion_matrix.tn)

    @property
    def true_negative_rate(self):
        '''
        Convenience property for the True Negative Rate (TN/FP+TN)
        '''
        return self.confusion_matrix.tn / (self.confusion_matrix.fp + self.confusion_matrix.tn)

    @property
    def false_negative_rate(self):
        '''
        Convenience property for the False Negative Rate (FN/TP+FN)
        '''
        return self.confusion_matrix.fn / (self.confusion_matrix.tp + self.confusion_matrix.fn)

    @property
    def false_discovery_rate(self):
        '''
        Convenience property for the False Discovery Rate (FP/FP+TP)
        '''
        return self.confusion_matrix.fp / (self.confusion_matrix.fp + self.confusion_matrix.tp)

    @property
    def false_omission_rate(self):
        '''
        Convenience property for the False Omission Rate (FN/TN+FN)
        '''
        return self.confusion_matrix.fn / (self.confusion_matrix.tn + self.confusion_matrix.fn)

    @property
    def positive_predictive_value(self):
        '''
        Convenience property for the Positive Predictive Value (TP/FP+TP)
        '''
        return self.confusion_matrix.tp / (self.confusion_matrix.fp + self.confusion_matrix.tp)

    @property
    def negative_predictive_value(self):
        '''
        Convenience property for the Negative Predictive Value (TN/TN+FN)
        '''
        return self.confusion_matrix.tn / (self.confusion_matrix.tn + self.confusion_matrix.fn)

    @property
    def predicted_positive_rate(self):
        '''
        Convenience property for the Predicted Positive Rate (TP+FP/TP+FP+TN+FN)
        '''
        return (self.confusion_matrix.tp + self.confusion_matrix.fp) /\
            (self.confusion_matrix.fp + self.confusion_matrix.tn + self.confusion_matrix.tp + self.confusion_matrix.fn)

    @property
    def predicted_negative_rate(self):
        '''
        Convenience property for the Predicted Negative Rate (TN+FN/TP+FP+TN+FN)
        '''
        return (self.confusion_matrix.tn + self.confusion_matrix.fn) /\
            (self.confusion_matrix.fp + self.confusion_matrix.tn + self.confusion_matrix.tp + self.confusion_matrix.fn)

    @property
    def accuracy(self):
        '''
        Convenience property for the Accuracy Rate (TP+TN/TP+FP+TN+FN)
        '''
        return (self.confusion_matrix.tp + self.confusion_matrix.tn) /\
            (self.confusion_matrix.fp + self.confusion_matrix.tn + self.confusion_matrix.tp + self.confusion_matrix.fn)

    @property
    def f1(self):
        '''
        Convenience property for the F1 Score (2*TP/2*TP+FP+FN)
        '''
        return (2.0 * self.confusion_matrix.tp) / (2.0 * self.confusion_matrix.tp + self.confusion_matrix.fp + self.confusion_matrix.fn)

    @property
    def matthews_correlation_coefficient(self):
        '''
        Convenience property for the Matthews Correlation Coefficient (TP*TN-FP*FN/((FP+TP)*(TP+FN)*(TN+FP)*(TN+FN))^0.5)
        '''
        numerator = (self.confusion_matrix.tp * self.confusion_matrix.tn - self.confusion_matrix.fp * self.confusion_matrix.fn)
        denominator = (
            (self.confusion_matrix.fp + self.confusion_matrix.tp)
            * (self.confusion_matrix.tp + self.confusion_matrix.fn)
            * (self.confusion_matrix.tn + self.confusion_matrix.fp)
            * (self.confusion_matrix.tn + self.confusion_matrix.fn)
        )**0.5
        return numerator / denominator

    @property
    def informedness(self):
        '''
        Convenience property for the Informedness (TPR+TNR-1)
        '''
        return self.true_positive_rate + self.true_negative_rate - 1

    @property
    def markedness(self):
        '''
        Convenience property for the Markedness (PPV+NPV-1)
        '''
        return self.positive_predictive_value + self.negative_predictive_value - 1


############################### AGGREGATE METRICS ###############################

'''
Pointwise metrics using only the predict scoring method
(fixed operating point)
'''


class AggregateBinaryClassificationMetric(BinaryClassificationMetric):
    @staticmethod
    @abstractmethod
    def _score(predictions, labels):
        '''
        Each aggregate needs to define a separate private method to actually
        calculate the aggregate

        Separated from the public score method to enable easier testing and
        extension (values can be passed from non internal properties)
        '''

    def score(self):
        '''
        Main scoring method. Uses internal values and passes to class level
        aggregation method
        '''
        predictions = self.predictions
        labels = self.labels
        self.values = {'agg': self._score(predictions, labels)}


class AccuracyMetric(AggregateBinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'classification_accuracy'
        super(AccuracyMetric, self).__init__(**kwargs)

    @staticmethod
    def _score(predictions, labels):
        return accuracy_score(y_true=labels, y_pred=predictions)


class TprMetric(AggregateBinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr'
        super(TprMetric, self).__init__(**kwargs)

    @staticmethod
    def _score(predictions, labels):
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        return float(tp) / (tp + fn)


class FprMetric(AggregateBinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr'
        super(FprMetric, self).__init__(**kwargs)

    @staticmethod
    def _score(predictions, labels):
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        return float(fp) / (fp + tn)


class F1ScoreMetric(AggregateBinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'f1_score'
        super(F1ScoreMetric, self).__init__(**kwargs)

    @staticmethod
    def _score(predictions, labels):
        return f1_score(y_true=labels, y_pred=predictions)


'''
Aggregate metrics computed by evaluating over entire curves
(Requires proba method)
'''


class RocAucMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'roc_auc'
        super(RocAucMetric, self).__init__(**kwargs)

    @staticmethod
    def _score(probabilities, labels):
        return roc_auc_score(y_true=labels, y_score=probabilities)

    def score(self):
        probabilities = self.probabilities
        labels = self.labels
        self.values = {'agg': self._score(probabilities, labels)}

############################### CURVE METRICS ###############################


'''
Threshold Constrained Metrics
'''


class ThresholdTprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_tpr_curve'
        super(ThresholdTprMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.thresholds, self.true_positive_rate, maximize=True)


class ThresholdTnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_tnr_curve'
        super(ThresholdTnrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.thresholds, self.true_negative_rate, maximize=True)


class ThresholdFnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_fnr_curve'
        super(ThresholdFnrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.thresholds, self.false_negative_rate, maximize=False)


class ThresholdFprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_fpr_curve'
        super(ThresholdFprMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.thresholds, self.false_positive_rate, maximize=False)


class ThresholdFdrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_fdr_curve'
        super(ThresholdFdrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.thresholds, self.false_discovery_rate, maximize=False)


class ThresholdForMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_for_curve'
        super(ThresholdForMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.thresholds, self.false_omission_rate, maximize=False)


class ThresholdPpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_ppv_curve'
        super(ThresholdPpvMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.thresholds, self.positive_predictive_value, maximize=True)


class ThresholdNpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_npv_curve'
        super(ThresholdNpvMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.thresholds, self.negative_predictive_value, maximize=True)


class ThresholdPredictedPositiveRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_predicted_positive_rate_curve'
        super(ThresholdPredictedPositiveRateMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.thresholds, self.predicted_positive_rate, maximize=True)


class ThresholdPredictedNegativeRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_predicted_negative_rate_curve'
        super(ThresholdPredictedNegativeRateMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.thresholds, self.predicted_negative_rate, maximize=True)


class ThresholdAccuracyMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_accuracy_curve'
        super(ThresholdAccuracyMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.thresholds, self.accuracy, maximize=True)


class ThresholdF1ScoreMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_f1_score_curve'
        super(ThresholdF1ScoreMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.thresholds, self.f1, maximize=True)


class ThresholdMccMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_mcc_curve'
        super(ThresholdMccMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.thresholds, self.matthews_correlation_coefficient, maximize=True)


class ThresholdInformednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_informedness_curve'
        super(ThresholdInformednessMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.thresholds, self.informedness, maximize=True)


class ThresholdMarkednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_markedness_curve'
        super(ThresholdMarkednessMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.thresholds, self.markedness, maximize=True)


'''
FPR Constrained Metrics
'''


class FprThresholdMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_threshold_curve'
        super(FprThresholdMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_positive_rate, self.thresholds, maximize=False)


class FprTprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_tpr_curve'
        super(FprTprMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_positive_rate, self.true_positive_rate, maximize=True)


class FprTnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_tnr_curve'
        super(FprTnrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_positive_rate, self.true_negative_rate, maximize=True)


class FprFnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_fnr_curve'
        super(FprFnrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_positive_rate, self.false_negative_rate, maximize=False)


class FprFdrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_fdr_curve'
        super(FprFdrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_positive_rate, self.false_discovery_rate, maximize=False)


class FprForMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_for_curve'
        super(FprForMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_positive_rate, self.false_omission_rate, maximize=False)


class FprPpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_ppv_curve'
        super(FprPpvMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_positive_rate, self.positive_predictive_value, maximize=True)


class FprNpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_npv_curve'
        super(FprNpvMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_positive_rate, self.negative_predictive_value, maximize=True)


class FprPredictedPositiveRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_predicted_positive_rate_curve'
        super(FprPredictedPositiveRateMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_positive_rate, self.predicted_positive_rate, maximize=True)


class FprPredictedNegativeRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_predicted_negative_rate_curve'
        super(FprPredictedNegativeRateMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_positive_rate, self.predicted_negative_rate, maximize=True)


class FprAccuracyMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_accuracy_curve'
        super(FprAccuracyMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_positive_rate, self.accuracy, maximize=True)


class FprF1ScoreMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_f1_score_curve'
        super(FprF1ScoreMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_positive_rate, self.f1, maximize=True)


class FprMccMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_mcc_curve'
        super(FprMccMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_positive_rate, self.matthews_correlation_coefficient, maximize=True)


class FprInformednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_informedness_curve'
        super(FprInformednessMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_positive_rate, self.informedness, maximize=True)


class FprMarkednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_markedness_curve'
        super(FprMarkednessMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_positive_rate, self.markedness, maximize=True)


'''
TPR Constrained Metrics
'''


class TprThresholdMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_threshold_curve'
        super(TprThresholdMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_positive_rate, self.thresholds, maximize=False)


class TprFprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_fpr_curve'
        super(TprFprMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_positive_rate, self.false_positive_rate, maximize=True)


class TprTnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_tnr_curve'
        super(TprTnrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_positive_rate, self.true_negative_rate, maximize=True)


class TprFnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_fnr_curve'
        super(TprFnrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_positive_rate, self.false_negative_rate, maximize=False)


class TprFdrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_fdr_curve'
        super(TprFdrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_positive_rate, self.false_discovery_rate, maximize=False)


class TprForMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_for_curve'
        super(TprForMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_positive_rate, self.false_omission_rate, maximize=False)


class TprPpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_ppv_curve'
        super(TprPpvMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_positive_rate, self.positive_predictive_value, maximize=True)


class TprNpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_npv_curve'
        super(TprNpvMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_positive_rate, self.negative_predictive_value, maximize=True)


class TprPredictedPositiveRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_predicted_positive_rate_curve'
        super(TprPredictedPositiveRateMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_positive_rate, self.predicted_positive_rate, maximize=True)


class TprPredictedNegativeRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_predicted_negative_rate_curve'
        super(TprPredictedNegativeRateMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_positive_rate, self.predicted_negative_rate, maximize=True)


class TprAccuracyMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_accuracy_curve'
        super(TprAccuracyMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_positive_rate, self.accuracy, maximize=True)


class TprF1ScoreMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_f1_score_curve'
        super(TprF1ScoreMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_positive_rate, self.f1, maximize=True)


class TprMccMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_mcc_curve'
        super(TprMccMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_positive_rate, self.matthews_correlation_coefficient, maximize=True)


class TprInformednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_informedness_curve'
        super(TprInformednessMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_positive_rate, self.informedness, maximize=True)


class TprMarkednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_markedness_curve'
        super(TprMarkednessMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_positive_rate, self.markedness, maximize=True)


'''
TNR Constrained Metrics
'''


class TnrThresholdMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tnr_threshold_curve'
        super(TnrThresholdMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_negative_rate, self.thresholds, maximize=False)


class TnrFprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tnr_fpr_curve'
        super(TnrFprMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_negative_rate, self.false_positive_rate, maximize=True)


class TnrTprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tnr_tpr_curve'
        super(TnrTprMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_negative_rate, self.true_positive_rate, maximize=True)


class TnrFnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tnr_fnr_curve'
        super(TnrFnrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_negative_rate, self.false_negative_rate, maximize=False)


class TnrFdrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tnr_fdr_curve'
        super(TnrFdrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_negative_rate, self.false_discovery_rate, maximize=False)


class TnrForMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tnr_for_curve'
        super(TnrForMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_negative_rate, self.false_omission_rate, maximize=False)


class TnrPpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tnr_ppv_curve'
        super(TnrPpvMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_negative_rate, self.positive_predictive_value, maximize=True)


class TnrNpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tnr_npv_curve'
        super(TnrNpvMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_negative_rate, self.negative_predictive_value, maximize=True)


class TnrPredictedPositiveRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tnr_predicted_positive_rate_curve'
        super(TnrPredictedPositiveRateMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_negative_rate, self.predicted_positive_rate, maximize=True)


class TnrPredictedNegativeRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tnr_predicted_negative_rate_curve'
        super(TnrPredictedNegativeRateMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_negative_rate, self.predicted_negative_rate, maximize=True)


class TnrAccuracyMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tnr_accuracy_curve'
        super(TnrAccuracyMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_negative_rate, self.accuracy, maximize=True)


class TnrF1ScoreMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tnr_f1_score_curve'
        super(TnrF1ScoreMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_negative_rate, self.f1, maximize=True)


class TnrMccMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tnr_mcc_curve'
        super(TnrMccMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_negative_rate, self.matthews_correlation_coefficient, maximize=True)


class TnrInformednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tnr_informedness_curve'
        super(TnrInformednessMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_negative_rate, self.informedness, maximize=True)


class TnrMarkednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tnr_markedness_curve'
        super(TnrMarkednessMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.true_negative_rate, self.markedness, maximize=True)


'''
FNR Constrained Metrics
'''


class FnrThresholdMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fnr_threshold_curve'
        super(FnrThresholdMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_negative_rate, self.thresholds, maximize=False)


class FnrFprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fnr_fpr_curve'
        super(FnrFprMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_negative_rate, self.false_positive_rate, maximize=True)


class FnrTprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fnr_tpr_curve'
        super(FnrTprMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_negative_rate, self.true_positive_rate, maximize=True)


class FnrTnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fnr_tnr_curve'
        super(FnrTnrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_negative_rate, self.true_negative_rate, maximize=True)


class FnrFdrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fnr_fdr_curve'
        super(FnrFdrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_negative_rate, self.false_discovery_rate, maximize=False)


class FnrForMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fnr_for_curve'
        super(FnrForMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_negative_rate, self.false_omission_rate, maximize=False)


class FnrPpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fnr_ppv_curve'
        super(FnrPpvMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_negative_rate, self.positive_predictive_value, maximize=True)


class FnrNpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fnr_npv_curve'
        super(FnrNpvMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_negative_rate, self.negative_predictive_value, maximize=True)


class FnrPredictedPositiveRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fnr_predicted_positive_rate_curve'
        super(FnrPredictedPositiveRateMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_negative_rate, self.predicted_positive_rate, maximize=True)


class FnrPredictedNegativeRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fnr_predicted_negative_rate_curve'
        super(FnrPredictedNegativeRateMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_negative_rate, self.predicted_negative_rate, maximize=True)


class FnrAccuracyMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fnr_accuracy_curve'
        super(FnrAccuracyMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_negative_rate, self.accuracy, maximize=True)


class FnrF1ScoreMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fnr_f1_score_curve'
        super(FnrF1ScoreMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_negative_rate, self.f1, maximize=True)


class FnrMccMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fnr_mcc_curve'
        super(FnrMccMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_negative_rate, self.matthews_correlation_coefficient, maximize=True)


class FnrInformednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fnr_informedness_curve'
        super(FnrInformednessMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_negative_rate, self.informedness, maximize=True)


class FnrMarkednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fnr_markedness_curve'
        super(FnrMarkednessMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_negative_rate, self.markedness, maximize=True)


'''
FDR Constrained Metrics
'''


class FdrThresholdMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fdr_threshold_curve'
        super(FdrThresholdMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_discovery_rate, self.thresholds, maximize=False)


class FdrFprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fdr_fpr_curve'
        super(FdrFprMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_discovery_rate, self.false_positive_rate, maximize=True)


class FdrTprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fdr_tpr_curve'
        super(FdrTprMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_discovery_rate, self.true_positive_rate, maximize=True)


class FdrTnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fdr_tnr_curve'
        super(FdrTnrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_discovery_rate, self.true_negative_rate, maximize=True)


class FdrFnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fdr_fnr_curve'
        super(FdrFnrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_discovery_rate, self.false_negative_rate, maximize=False)


class FdrForMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fdr_for_curve'
        super(FdrForMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_discovery_rate, self.false_omission_rate, maximize=False)


class FdrPpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fdr_ppv_curve'
        super(FdrPpvMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_discovery_rate, self.positive_predictive_value, maximize=True)


class FdrNpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fdr_npv_curve'
        super(FdrNpvMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_discovery_rate, self.negative_predictive_value, maximize=True)


class FdrPredictedPositiveRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fdr_predicted_positive_rate_curve'
        super(FdrPredictedPositiveRateMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_discovery_rate, self.predicted_positive_rate, maximize=True)


class FdrPredictedNegativeRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fdr_predicted_negative_rate_curve'
        super(FdrPredictedNegativeRateMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_discovery_rate, self.predicted_negative_rate, maximize=True)


class FdrAccuracyMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fdr_accuracy_curve'
        super(FdrAccuracyMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_discovery_rate, self.accuracy, maximize=True)


class FdrF1ScoreMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fdr_f1_score_curve'
        super(FdrF1ScoreMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_discovery_rate, self.f1, maximize=True)


class FdrMccMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fdr_mcc_curve'
        super(FdrMccMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_discovery_rate, self.matthews_correlation_coefficient, maximize=True)


class FdrInformednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fdr_informedness_curve'
        super(FdrInformednessMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_discovery_rate, self.informedness, maximize=True)


class FdrMarkednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fdr_markedness_curve'
        super(FdrMarkednessMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_discovery_rate, self.markedness, maximize=True)


'''
FOR Constrained Metrics
'''


class ForThresholdMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'for_threshold_curve'
        super(ForThresholdMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_omission_rate, self.thresholds, maximize=False)


class ForFprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'for_fpr_curve'
        super(ForFprMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_omission_rate, self.false_positive_rate, maximize=True)


class ForTprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'for_tpr_curve'
        super(ForTprMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_omission_rate, self.true_positive_rate, maximize=True)


class ForTnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'for_tnr_curve'
        super(ForTnrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_omission_rate, self.true_negative_rate, maximize=True)


class ForFnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'for_fnr_curve'
        super(ForFnrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_omission_rate, self.false_negative_rate, maximize=False)


class ForFdrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'for_fdr_curve'
        super(ForFdrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_omission_rate, self.false_discovery_rate, maximize=False)


class ForPpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'for_ppv_curve'
        super(ForPpvMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_omission_rate, self.positive_predictive_value, maximize=True)


class ForNpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'for_npv_curve'
        super(ForNpvMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_omission_rate, self.negative_predictive_value, maximize=True)


class ForPredictedPositiveRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'for_predicted_positive_rate_curve'
        super(ForPredictedPositiveRateMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_omission_rate, self.predicted_positive_rate, maximize=True)


class ForPredictedNegativeRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'for_predicted_negative_rate_curve'
        super(ForPredictedNegativeRateMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_omission_rate, self.predicted_negative_rate, maximize=True)


class ForAccuracyMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'for_accuracy_curve'
        super(ForAccuracyMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_omission_rate, self.accuracy, maximize=True)


class ForF1ScoreMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'for_f1_score_curve'
        super(ForF1ScoreMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_omission_rate, self.f1, maximize=True)


class ForMccMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'for_mcc_curve'
        super(ForMccMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_omission_rate, self.matthews_correlation_coefficient, maximize=True)


class ForInformednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'for_informedness_curve'
        super(ForInformednessMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_omission_rate, self.informedness, maximize=True)


class ForMarkednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'for_markedness_curve'
        super(ForMarkednessMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.false_omission_rate, self.markedness, maximize=True)


'''
PPV Constrained Metrics
'''


class PpvThresholdMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'ppv_threshold_curve'
        super(PpvThresholdMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.positive_predictive_value, self.thresholds, maximize=False)


class PpvFprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'ppv_fpr_curve'
        super(PpvFprMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.positive_predictive_value, self.false_positive_rate, maximize=True)


class PpvTprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'ppv_tpr_curve'
        super(PpvTprMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.positive_predictive_value, self.true_positive_rate, maximize=True)


class PpvTnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'ppv_tnr_curve'
        super(PpvTnrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.positive_predictive_value, self.true_negative_rate, maximize=True)


class PpvFnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'ppv_fnr_curve'
        super(PpvFnrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.positive_predictive_value, self.false_negative_rate, maximize=False)


class PpvFdrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'ppv_fdr_curve'
        super(PpvFdrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.positive_predictive_value, self.false_discovery_rate, maximize=False)


class PpvForMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'ppv_for_curve'
        super(PpvForMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.positive_predictive_value, self.false_omission_rate, maximize=False)


class PpvNpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'ppv_npv_curve'
        super(PpvNpvMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.positive_predictive_value, self.negative_predictive_value, maximize=True)


class PpvPredictedPositiveRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'ppv_predicted_positive_rate_curve'
        super(PpvPredictedPositiveRateMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.positive_predictive_value, self.predicted_positive_rate, maximize=True)


class PpvPredictedNegativeRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'ppv_predicted_negative_rate_curve'
        super(PpvPredictedNegativeRateMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.positive_predictive_value, self.predicted_negative_rate, maximize=True)


class PpvAccuracyMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'ppv_accuracy_curve'
        super(PpvAccuracyMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.positive_predictive_value, self.accuracy, maximize=True)


class PpvF1ScoreMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'ppv_f1_score_curve'
        super(PpvF1ScoreMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.positive_predictive_value, self.f1, maximize=True)


class PpvMccMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'ppv_mcc_curve'
        super(PpvMccMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.positive_predictive_value, self.matthews_correlation_coefficient, maximize=True)


class PpvInformednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'ppv_informedness_curve'
        super(PpvInformednessMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.positive_predictive_value, self.informedness, maximize=True)


class PpvMarkednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'ppv_markedness_curve'
        super(PpvMarkednessMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.positive_predictive_value, self.markedness, maximize=True)


'''
NPV Constrained Metrics
'''


class NpvThresholdMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'npv_threshold_curve'
        super(NpvThresholdMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.negative_predictive_value, self.thresholds, maximize=False)


class NpvFprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'npv_fpr_curve'
        super(NpvFprMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.negative_predictive_value, self.false_positive_rate, maximize=True)


class NpvTprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'npv_tpr_curve'
        super(NpvTprMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.negative_predictive_value, self.true_positive_rate, maximize=True)


class NpvTnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'npv_tnr_curve'
        super(NpvTnrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.negative_predictive_value, self.true_negative_rate, maximize=True)


class NpvFnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'npv_fnr_curve'
        super(NpvFnrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.negative_predictive_value, self.false_negative_rate, maximize=False)


class NpvFdrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'npv_fdr_curve'
        super(NpvFdrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.negative_predictive_value, self.false_discovery_rate, maximize=False)


class NpvForMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'npv_for_curve'
        super(NpvForMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.negative_predictive_value, self.false_omission_rate, maximize=False)


class NpvPpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'npv_ppv_curve'
        super(NpvPpvMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.negative_predictive_value, self.positive_predictive_value, maximize=True)


class NpvPredictedPositiveRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'npv_predicted_positive_rate_curve'
        super(NpvPredictedPositiveRateMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.negative_predictive_value, self.predicted_positive_rate, maximize=True)


class NpvPredictedNegativeRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'npv_predicted_negative_rate_curve'
        super(NpvPredictedNegativeRateMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.negative_predictive_value, self.predicted_negative_rate, maximize=True)


class NpvAccuracyMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'npv_accuracy_curve'
        super(NpvAccuracyMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.negative_predictive_value, self.accuracy, maximize=True)


class NpvF1ScoreMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'npv_f1_score_curve'
        super(NpvF1ScoreMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.negative_predictive_value, self.f1, maximize=True)


class NpvMccMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'npv_mcc_curve'
        super(NpvMccMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.negative_predictive_value, self.matthews_correlation_coefficient, maximize=True)


class NpvInformednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'npv_informedness_curve'
        super(NpvInformednessMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.negative_predictive_value, self.informedness, maximize=True)


class NpvMarkednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'npv_markedness_curve'
        super(NpvMarkednessMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.negative_predictive_value, self.markedness, maximize=True)


'''
Predicted Positive Rate Constrained Metrics
'''


class PredictedPositiveRateThresholdMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_positive_rate_threshold_curve'
        super(PredictedPositiveRateThresholdMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_positive_rate, self.thresholds, maximize=False)


class PredictedPositiveRateFprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_positive_rate_fpr_curve'
        super(PredictedPositiveRateFprMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_positive_rate, self.false_positive_rate, maximize=True)


class PredictedPositiveRateTprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_positive_rate_tpr_curve'
        super(PredictedPositiveRateTprMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_positive_rate, self.true_positive_rate, maximize=True)


class PredictedPositiveRateTnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_positive_rate_tnr_curve'
        super(PredictedPositiveRateTnrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_positive_rate, self.true_negative_rate, maximize=True)


class PredictedPositiveRateFnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_positive_rate_fnr_curve'
        super(PredictedPositiveRateFnrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_positive_rate, self.false_negative_rate, maximize=False)


class PredictedPositiveRateFdrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_positive_rate_fdr_curve'
        super(PredictedPositiveRateFdrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_positive_rate, self.false_discovery_rate, maximize=False)


class PredictedPositiveRateForMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_positive_rate_for_curve'
        super(PredictedPositiveRateForMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_positive_rate, self.false_omission_rate, maximize=False)


class PredictedPositiveRatePpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_positive_rate_ppv_curve'
        super(PredictedPositiveRatePpvMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_positive_rate, self.positive_predictive_value, maximize=True)


class PredictedPositiveRateNpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_positive_rate_npv_curve'
        super(PredictedPositiveRateNpvMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_positive_rate, self.negative_predictive_value, maximize=True)


class PredictedPositiveRatePredictedNegativeRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_positive_rate_predicted_negative_rate_curve'
        super(PredictedPositiveRatePredictedNegativeRateMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_positive_rate, self.predicted_negative_rate, maximize=True)


class PredictedPositiveRateAccuracyMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_positive_rate_accuracy_curve'
        super(PredictedPositiveRateAccuracyMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_positive_rate, self.accuracy, maximize=True)


class PredictedPositiveRateF1ScoreMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_positive_rate_f1_score_curve'
        super(PredictedPositiveRateF1ScoreMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_positive_rate, self.f1, maximize=True)


class PredictedPositiveRateMccMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_positive_rate_mcc_curve'
        super(PredictedPositiveRateMccMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_positive_rate, self.matthews_correlation_coefficient, maximize=True)


class PredictedPositiveRateInformednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_positive_rate_informedness_curve'
        super(PredictedPositiveRateInformednessMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_positive_rate, self.informedness, maximize=True)


class PredictedPositiveRateMarkednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_positive_rate_markedness_curve'
        super(PredictedPositiveRateMarkednessMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_positive_rate, self.markedness, maximize=True)


'''
Predicted Negative Rate Constrained Metrics
'''


class PredictedNegativeRateThresholdMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_negative_rate_threshold_curve'
        super(PredictedNegativeRateThresholdMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_negative_rate, self.thresholds, maximize=False)


class PredictedNegativeRateFprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_negative_rate_fpr_curve'
        super(PredictedNegativeRateFprMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_negative_rate, self.false_positive_rate, maximize=True)


class PredictedNegativeRateTprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_negative_rate_tpr_curve'
        super(PredictedNegativeRateTprMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_negative_rate, self.true_positive_rate, maximize=True)


class PredictedNegativeRateTnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_negative_rate_tnr_curve'
        super(PredictedNegativeRateTnrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_negative_rate, self.true_negative_rate, maximize=True)


class PredictedNegativeRateFnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_negative_rate_fnr_curve'
        super(PredictedNegativeRateFnrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_negative_rate, self.false_negative_rate, maximize=False)


class PredictedNegativeRateFdrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_negative_rate_fdr_curve'
        super(PredictedNegativeRateFdrMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_negative_rate, self.false_discovery_rate, maximize=False)


class PredictedNegativeRateForMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_negative_rate_for_curve'
        super(PredictedNegativeRateForMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_negative_rate, self.false_omission_rate, maximize=False)


class PredictedNegativeRatePpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_negative_rate_ppv_curve'
        super(PredictedNegativeRatePpvMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_negative_rate, self.positive_predictive_value, maximize=True)


class PredictedNegativeRateNpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_negative_rate_npv_curve'
        super(PredictedNegativeRateNpvMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_negative_rate, self.negative_predictive_value, maximize=True)


class PredictedNegativeRatePredictedPositiveRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_negative_rate_predicted_positive_rate_curve'
        super(PredictedNegativeRatePredictedPositiveRateMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_negative_rate, self.predicted_positive_rate, maximize=True)


class PredictedNegativeRateAccuracyMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_negative_rate_accuracy_curve'
        super(PredictedNegativeRateAccuracyMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_negative_rate, self.accuracy, maximize=True)


class PredictedNegativeRateF1ScoreMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_negative_rate_f1_score_curve'
        super(PredictedNegativeRateF1ScoreMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_negative_rate, self.f1, maximize=True)


class PredictedNegativeRateMccMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_negative_rate_mcc_curve'
        super(PredictedNegativeRateMccMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_negative_rate, self.matthews_correlation_coefficient, maximize=True)


class PredictedNegativeRateInformednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_negative_rate_informedness_curve'
        super(PredictedNegativeRateInformednessMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_negative_rate, self.informedness, maximize=True)


class PredictedNegativeRateMarkednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'predicted_negative_rate_markedness_curve'
        super(PredictedNegativeRateMarkednessMetric, self).__init__(**kwargs)

    def score(self):
        self.values = self.dedupe_curve(self.predicted_negative_rate, self.markedness, maximize=True)
