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
from simpleml import TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score
import numpy as np
import pandas as pd


__author__ = 'Elisha Yadgaran'


############################### BASE ###############################

class ClassificationMetric(Metric):
    '''
    TODO: Figure out multiclass generalizations
    '''
    def __init__(self, dataset_split, **kwargs):
        '''
        :param dataset_split: string denoting which dataset split to use
            can be one of: `TRAIN`, `VALIDATION`, Other. Other gets no prefix
            Default is train split to stay consistent with no split mapping to Train
            in Pipeline

        '''
        name = kwargs.pop('name', '')
        self.dataset_split = dataset_split

        # Explicitly call out in sample or validation metrics
        if dataset_split == TRAIN_SPLIT:
            name = 'in_sample_' + name
        elif dataset_split == VALIDATION_SPLIT:
            name = 'validation_' + name

        super(ClassificationMetric, self).__init__(name=name, **kwargs)

    @property
    def labels(self):
        return self.model.get_labels(dataset_split=self.dataset_split)

    @property
    def probabilities(self):
        return self.model.predict_proba(X=None, dataset_split=self.dataset_split)

    @property
    def predictions(self):
        return self.model.predict(X=None, dataset_split=self.dataset_split)


class BinaryClassificationMetric(ClassificationMetric):
    def __init__(self, **kwargs):
        super(BinaryClassificationMetric, self).__init__(**kwargs)

        # Initialize confusion matrix
        self._confusion_matrix = None

        # Thresholds to compute confusion matrix at (default every 0.005 increment)
        self.thresholds = np.linspace(0, 1, 201)

    @property
    def probabilities(self):
        probabilities = self.model.predict_proba(X=None, dataset_split=self.dataset_split)
        if len(probabilities.shape) > 1 and probabilities.shape[1] > 1:
            # Indicates multiple class probabilities are returned (class_0, class_1)
            probabilities = probabilities[:, 1]
        return probabilities

    @property
    def predictions(self):
        predictions = self.model.predict(X=None, dataset_split=self.dataset_split)
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
        if self._confusion_matrix is None:
            self.create_confusion_matrix()

        return self._confusion_matrix

    def create_confusion_matrix(self):
        '''
        Iterate through each threshold and compute confusion matrix
        '''
        probabilities = self.probabilities
        labels = self.labels

        results = []
        for threshold in self.thresholds:
            predictions = np.where(probabilities >= threshold, 1, 0)
            tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
            results.append((threshold, tn, fp, fn, tp))

        self._confusion_matrix = pd.DataFrame(results, columns=['threshold', 'tn', 'fp', 'fn', 'tp'])

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

        df = pd.DataFrame(zip(keys, values), columns=['keys', 'values'])

        agg = 'max' if maximize else 'min'
        return df.groupby('keys').agg({'values': agg}).to_dict()['values']


############################### AGGREGATE METRICS ###############################

'''
Pointwise metrics using only the predict scoring method
(fixed operating point)
'''

class AccuracyMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'classification_accuracy'
        super(AccuracyMetric, self).__init__(**kwargs)

    def score(self):
        predictions = self.predictions
        labels = self.labels
        accuracy = accuracy_score(y_true=labels, y_pred=predictions)

        self.values = {'agg': accuracy}

class TprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr'
        super(TprMetric, self).__init__(**kwargs)

    def score(self):
        predictions = self.predictions
        labels = self.labels
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        tpr = float(tp) / (tp + fn)

        self.values = {'agg': tpr}

class FprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr'
        super(FprMetric, self).__init__(**kwargs)

    def score(self):
        predictions = self.predictions
        labels = self.labels
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        fpr = float(fp) / (fp + tn)

        self.values = {'agg': fpr}

class F1ScoreMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'f1_score'
        super(F1ScoreMetric, self).__init__(**kwargs)

    def score(self):
        predictions = self.predictions
        labels = self.labels
        f1_score_ = f1_score(y_true=labels, y_pred=predictions)

        self.values = {'agg': f1_score_}


'''
Aggregate metrics computed by evaluating over entire curves
(Requires proba method)
'''

class RocAucMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'roc_auc'
        super(RocAucMetric, self).__init__(**kwargs)

    def score(self):
        probabilities = self.probabilities
        labels = self.labels
        auc = roc_auc_score(y_true=labels, y_score=probabilities)

        self.values = {'agg': auc}

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
        tpr = self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)
        thresholds = self.confusion_matrix.threshold

        self.values = self.dedupe_curve(thresholds, tpr, maximize=True)


class ThresholdTnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_tnr_curve'
        super(ThresholdTnrMetric, self).__init__(**kwargs)

    def score(self):
        tnr = self.confusion_matrix.tn / (self.confusion_matrix.fp + self.confusion_matrix.tn)
        thresholds = self.confusion_matrix.threshold

        self.values = self.dedupe_curve(thresholds, tnr, maximize=True)

class ThresholdFnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_fnr_curve'
        super(ThresholdFnrMetric, self).__init__(**kwargs)

    def score(self):
        fnr = self.confusion_matrix.fn / (self.confusion_matrix.tp + self.confusion_matrix.fn)
        thresholds = self.confusion_matrix.threshold

        self.values = self.dedupe_curve(thresholds, fnr, maximize=False)


class ThresholdFprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_fpr_curve'
        super(ThresholdFprMetric, self).__init__(**kwargs)

    def score(self):
        fpr = self.confusion_matrix.fp / (self.confusion_matrix.fp + self.confusion_matrix.tn)
        thresholds = self.confusion_matrix.threshold

        self.values = self.dedupe_curve(thresholds, fpr, maximize=False)


class ThresholdFdrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_fdr_curve'
        super(ThresholdFdrMetric, self).__init__(**kwargs)

    def score(self):
        fdr = self.confusion_matrix.fp / (self.confusion_matrix.fp + self.confusion_matrix.tp)
        thresholds = self.confusion_matrix.threshold

        self.values = self.dedupe_curve(thresholds, fdr, maximize=False)


class ThresholdForMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_for_curve'
        super(ThresholdForMetric, self).__init__(**kwargs)

    def score(self):
        false_omission_rate = self.confusion_matrix.fn / (self.confusion_matrix.tn + self.confusion_matrix.fn)
        thresholds = self.confusion_matrix.threshold

        self.values = self.dedupe_curve(thresholds, false_omission_rate, maximize=False)


class ThresholdPpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_ppv_curve'
        super(ThresholdPpvMetric, self).__init__(**kwargs)

    def score(self):
        ppv = self.confusion_matrix.tp / (self.confusion_matrix.fp + self.confusion_matrix.tp)
        thresholds = self.confusion_matrix.threshold

        self.values = self.dedupe_curve(thresholds, ppv, maximize=True)


class ThresholdNpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_npv_curve'
        super(ThresholdNpvMetric, self).__init__(**kwargs)

    def score(self):
        npv = self.confusion_matrix.tn / (self.confusion_matrix.tn + self.confusion_matrix.fn)
        thresholds = self.confusion_matrix.threshold

        self.values = self.dedupe_curve(thresholds, npv, maximize=True)


class ThresholdPredictedPositiveRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_predicted_positive_rate_curve'
        super(ThresholdPredictedPositiveRateMetric, self).__init__(**kwargs)

    def score(self):
        ppr = (self.confusion_matrix.tp + self.confusion_matrix.fp) /\
            (self.confusion_matrix.fp + self.confusion_matrix.tn + self.confusion_matrix.tp + self.confusion_matrix.fn)
        thresholds = self.confusion_matrix.threshold

        self.values = self.dedupe_curve(thresholds, ppr, maximize=True)


class ThresholdPredictedNegativeRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_predicted_negative_rate_curve'
        super(ThresholdPredictedNegativeRateMetric, self).__init__(**kwargs)

    def score(self):
        pnr = (self.confusion_matrix.tn + self.confusion_matrix.fn) /\
            (self.confusion_matrix.fp + self.confusion_matrix.tn + self.confusion_matrix.tp + self.confusion_matrix.fn)
        thresholds = self.confusion_matrix.threshold

        self.values = self.dedupe_curve(thresholds, pnr, maximize=True)


class ThresholdAccuracyMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_accuracy_curve'
        super(ThresholdAccuracyMetric, self).__init__(**kwargs)

    def score(self):
        accuracy = (self.confusion_matrix.tp + self.confusion_matrix.tn) /\
            (self.confusion_matrix.fp + self.confusion_matrix.tn + self.confusion_matrix.tp + self.confusion_matrix.fn)
        thresholds = self.confusion_matrix.threshold

        self.values = self.dedupe_curve(thresholds, accuracy, maximize=True)


class ThresholdF1ScoreMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_f1_score_curve'
        super(ThresholdF1ScoreMetric, self).__init__(**kwargs)

    def score(self):
        f1_score_ = (2.0 * self.confusion_matrix.tp) / (2.0 * self.confusion_matrix.tp + self.confusion_matrix.fp + self.confusion_matrix.fn)
        thresholds = self.confusion_matrix.threshold

        self.values = self.dedupe_curve(thresholds, f1_score_, maximize=True)


class ThresholdMccMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_mcc_curve'
        super(ThresholdMccMetric, self).__init__(**kwargs)

    def score(self):
        matthews_correlation_coefficient = (self.confusion_matrix.tp * self.confusion_matrix.tn - self.confusion_matrix.fp * self.confusion_matrix.fn) /\
            ((self.confusion_matrix.fp + self.confusion_matrix.tp) * (self.confusion_matrix.tp + self.confusion_matrix.fn) *\
             (self.confusion_matrix.tn + self.confusion_matrix.fp) * (self.confusion_matrix.tn + self.confusion_matrix.fn))**0.5
        thresholds = self.confusion_matrix.threshold

        self.values = self.dedupe_curve(thresholds, matthews_correlation_coefficient, maximize=True)


class ThresholdInformednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_informedness_curve'
        super(ThresholdInformednessMetric, self).__init__(**kwargs)

    def score(self):
        tpr = self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)
        tnr = self.confusion_matrix.tn / (self.confusion_matrix.fp + self.confusion_matrix.tn)

        informedness = tpr + tnr - 1
        thresholds = self.confusion_matrix.threshold

        self.values = self.dedupe_curve(thresholds, informedness, maximize=True)


class ThresholdMarkednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'threshold_markedness_curve'
        super(ThresholdMarkednessMetric, self).__init__(**kwargs)

    def score(self):
        ppv = self.confusion_matrix.tp / (self.confusion_matrix.fp + self.confusion_matrix.tp)
        npv = self.confusion_matrix.tn / (self.confusion_matrix.tn + self.confusion_matrix.fn)

        markedness = ppv + npv - 1
        thresholds = self.confusion_matrix.threshold

        self.values = self.dedupe_curve(thresholds, markedness, maximize=True)


'''
FPR Constrained Metrics
'''

class FprThresholdMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_threshold_curve'
        super(FprThresholdMetric, self).__init__(**kwargs)

    def score(self):
        fpr = self.confusion_matrix.fp / (self.confusion_matrix.fp + self.confusion_matrix.tn)
        thresholds = self.confusion_matrix.threshold

        self.values = self.dedupe_curve(fpr, thresholds, maximize=False)


class FprTprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_tpr_curve'
        super(FprTprMetric, self).__init__(**kwargs)

    def score(self):
        fpr = self.confusion_matrix.fp / (self.confusion_matrix.fp + self.confusion_matrix.tn)
        tpr = self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)

        self.values = self.dedupe_curve(fpr, tpr, maximize=True)


class FprTnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_tnr_curve'
        super(FprTnrMetric, self).__init__(**kwargs)

    def score(self):
        fpr = self.confusion_matrix.fp / (self.confusion_matrix.fp + self.confusion_matrix.tn)
        tnr = self.confusion_matrix.tn / (self.confusion_matrix.fp + self.confusion_matrix.tn)

        self.values = self.dedupe_curve(fpr, tnr, maximize=True)


class FprFnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_fnr_curve'
        super(FprFnrMetric, self).__init__(**kwargs)

    def score(self):
        fpr = self.confusion_matrix.fp / (self.confusion_matrix.fp + self.confusion_matrix.tn)
        fnr = self.confusion_matrix.fn / (self.confusion_matrix.tp + self.confusion_matrix.fn)

        self.values = self.dedupe_curve(fpr, fnr, maximize=False)


class FprFdrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_fdr_curve'
        super(FprFdrMetric, self).__init__(**kwargs)

    def score(self):
        fpr = self.confusion_matrix.fp / (self.confusion_matrix.fp + self.confusion_matrix.tn)
        fdr = self.confusion_matrix.fp / (self.confusion_matrix.fp + self.confusion_matrix.tp)

        self.values = self.dedupe_curve(fpr, fdr, maximize=False)


class FprForMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_for_curve'
        super(FprForMetric, self).__init__(**kwargs)

    def score(self):
        fpr = self.confusion_matrix.fp / (self.confusion_matrix.fp + self.confusion_matrix.tn)
        false_omission_rate = self.confusion_matrix.fn / (self.confusion_matrix.tn + self.confusion_matrix.fn)

        self.values = self.dedupe_curve(fpr, false_omission_rate, maximize=False)


class FprPpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_ppv_curve'
        super(FprPpvMetric, self).__init__(**kwargs)

    def score(self):
        fpr = self.confusion_matrix.fp / (self.confusion_matrix.fp + self.confusion_matrix.tn)
        ppv = self.confusion_matrix.tp / (self.confusion_matrix.fp + self.confusion_matrix.tp)

        self.values = self.dedupe_curve(fpr, ppv, maximize=True)


class FprNpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_npv_curve'
        super(FprNpvMetric, self).__init__(**kwargs)

    def score(self):
        fpr = self.confusion_matrix.fp / (self.confusion_matrix.fp + self.confusion_matrix.tn)
        npv = self.confusion_matrix.tn / (self.confusion_matrix.tn + self.confusion_matrix.fn)

        self.values = self.dedupe_curve(fpr, npv, maximize=True)


class FprPredictedPositiveRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_predicted_positive_rate_curve'
        super(FprPredictedPositiveRateMetric, self).__init__(**kwargs)

    def score(self):
        fpr = self.confusion_matrix.fp / (self.confusion_matrix.fp + self.confusion_matrix.tn)
        ppr = (self.confusion_matrix.tp + self.confusion_matrix.fp) /\
            (self.confusion_matrix.fp + self.confusion_matrix.tn + self.confusion_matrix.tp + self.confusion_matrix.fn)

        self.values = self.dedupe_curve(fpr, ppr, maximize=True)


class FprPredictedNegativeRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_predicted_negative_rate_curve'
        super(FprPredictedNegativeRateMetric, self).__init__(**kwargs)

    def score(self):
        fpr = self.confusion_matrix.fp / (self.confusion_matrix.fp + self.confusion_matrix.tn)
        pnr = (self.confusion_matrix.tn + self.confusion_matrix.fn) /\
            (self.confusion_matrix.fp + self.confusion_matrix.tn + self.confusion_matrix.tp + self.confusion_matrix.fn)

        self.values = self.dedupe_curve(fpr, pnr, maximize=True)


class FprAccuracyMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_accuracy_curve'
        super(FprAccuracyMetric, self).__init__(**kwargs)

    def score(self):
        fpr = self.confusion_matrix.fp / (self.confusion_matrix.fp + self.confusion_matrix.tn)
        accuracy = (self.confusion_matrix.tp + self.confusion_matrix.tn) /\
            (self.confusion_matrix.fp + self.confusion_matrix.tn + self.confusion_matrix.tp + self.confusion_matrix.fn)

        self.values = self.dedupe_curve(fpr, accuracy, maximize=True)


class FprF1ScoreMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_f1_score_curve'
        super(FprF1ScoreMetric, self).__init__(**kwargs)

    def score(self):
        fpr = self.confusion_matrix.fp / (self.confusion_matrix.fp + self.confusion_matrix.tn)
        f1_score_ = (2.0 * self.confusion_matrix.tp) / (2.0 * self.confusion_matrix.tp + self.confusion_matrix.fp + self.confusion_matrix.fn)

        self.values = self.dedupe_curve(fpr, f1_score_, maximize=True)


class FprMccMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_mcc_curve'
        super(FprMccMetric, self).__init__(**kwargs)

    def score(self):
        fpr = self.confusion_matrix.fp / (self.confusion_matrix.fp + self.confusion_matrix.tn)
        matthews_correlation_coefficient = (self.confusion_matrix.tp * self.confusion_matrix.tn - self.confusion_matrix.fp * self.confusion_matrix.fn) /\
            ((self.confusion_matrix.fp + self.confusion_matrix.tp) * (self.confusion_matrix.tp + self.confusion_matrix.fn) *\
             (self.confusion_matrix.tn + self.confusion_matrix.fp) * (self.confusion_matrix.tn + self.confusion_matrix.fn))**0.5

        self.values = self.dedupe_curve(fpr, matthews_correlation_coefficient, maximize=True)


class FprInformednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_informedness_curve'
        super(FprInformednessMetric, self).__init__(**kwargs)

    def score(self):
        fpr = self.confusion_matrix.fp / (self.confusion_matrix.fp + self.confusion_matrix.tn)
        tpr = self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)
        tnr = self.confusion_matrix.tn / (self.confusion_matrix.fp + self.confusion_matrix.tn)

        informedness = tpr + tnr - 1

        self.values = self.dedupe_curve(fpr, informedness, maximize=True)


class FprMarkednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'fpr_markedness_curve'
        super(FprMarkednessMetric, self).__init__(**kwargs)

    def score(self):
        fpr = self.confusion_matrix.fp / (self.confusion_matrix.fp + self.confusion_matrix.tn)
        ppv = self.confusion_matrix.tp / (self.confusion_matrix.fp + self.confusion_matrix.tp)
        npv = self.confusion_matrix.tn / (self.confusion_matrix.tn + self.confusion_matrix.fn)

        markedness = ppv + npv - 1

        self.values = self.dedupe_curve(fpr, markedness, maximize=True)


'''
TPR Constrained Metrics
'''

class TprThresholdMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_threshold_curve'
        super(TprThresholdMetric, self).__init__(**kwargs)

    def score(self):
        tpr = self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)
        thresholds = self.confusion_matrix.threshold

        self.values = self.dedupe_curve(tpr, thresholds, maximize=False)


class TprFprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_fpr_curve'
        super(TprFprMetric, self).__init__(**kwargs)

    def score(self):
        fpr = self.confusion_matrix.fp / (self.confusion_matrix.fp + self.confusion_matrix.tn)
        tpr = self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)

        self.values = self.dedupe_curve(tpr, fpr, maximize=True)


class TprTnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_tnr_curve'
        super(TprTnrMetric, self).__init__(**kwargs)

    def score(self):
        tpr = self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)
        tnr = self.confusion_matrix.tn / (self.confusion_matrix.fp + self.confusion_matrix.tn)

        self.values = self.dedupe_curve(tpr, tnr, maximize=True)


class TprFnrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_fnr_curve'
        super(TprFnrMetric, self).__init__(**kwargs)

    def score(self):
        tpr = self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)
        fnr = self.confusion_matrix.fn / (self.confusion_matrix.tp + self.confusion_matrix.fn)

        self.values = self.dedupe_curve(tpr, fnr, maximize=False)


class TprFdrMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_fdr_curve'
        super(TprFdrMetric, self).__init__(**kwargs)

    def score(self):
        tpr = self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)
        fdr = self.confusion_matrix.fp / (self.confusion_matrix.fp + self.confusion_matrix.tp)

        self.values = self.dedupe_curve(tpr, fdr, maximize=False)


class TprForMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_for_curve'
        super(TprForMetric, self).__init__(**kwargs)

    def score(self):
        tpr = self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)
        false_omission_rate = self.confusion_matrix.fn / (self.confusion_matrix.tn + self.confusion_matrix.fn)

        self.values = self.dedupe_curve(tpr, false_omission_rate, maximize=False)


class TprPpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_ppv_curve'
        super(TprPpvMetric, self).__init__(**kwargs)

    def score(self):
        tpr = self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)
        ppv = self.confusion_matrix.tp / (self.confusion_matrix.fp + self.confusion_matrix.tp)

        self.values = self.dedupe_curve(tpr, ppv, maximize=True)


class TprNpvMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_npv_curve'
        super(TprNpvMetric, self).__init__(**kwargs)

    def score(self):
        tpr = self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)
        npv = self.confusion_matrix.tn / (self.confusion_matrix.tn + self.confusion_matrix.fn)

        self.values = self.dedupe_curve(tpr, npv, maximize=True)


class TprPredictedPositiveRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_predicted_positive_rate_curve'
        super(TprPredictedPositiveRateMetric, self).__init__(**kwargs)

    def score(self):
        tpr = self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)
        ppr = (self.confusion_matrix.tp + self.confusion_matrix.fp) /\
            (self.confusion_matrix.fp + self.confusion_matrix.tn + self.confusion_matrix.tp + self.confusion_matrix.fn)

        self.values = self.dedupe_curve(tpr, ppr, maximize=True)


class TprPredictedNegativeRateMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_predicted_negative_rate_curve'
        super(TprPredictedNegativeRateMetric, self).__init__(**kwargs)

    def score(self):
        tpr = self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)
        pnr = (self.confusion_matrix.tn + self.confusion_matrix.fn) /\
            (self.confusion_matrix.fp + self.confusion_matrix.tn + self.confusion_matrix.tp + self.confusion_matrix.fn)

        self.values = self.dedupe_curve(tpr, pnr, maximize=True)


class TprAccuracyMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_accuracy_curve'
        super(TprAccuracyMetric, self).__init__(**kwargs)

    def score(self):
        tpr = self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)
        accuracy = (self.confusion_matrix.tp + self.confusion_matrix.tn) /\
            (self.confusion_matrix.fp + self.confusion_matrix.tn + self.confusion_matrix.tp + self.confusion_matrix.fn)

        self.values = self.dedupe_curve(tpr, accuracy, maximize=True)


class TprF1ScoreMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_f1_score_curve'
        super(TprF1ScoreMetric, self).__init__(**kwargs)

    def score(self):
        tpr = self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)
        f1_score_ = (2.0 * self.confusion_matrix.tp) / (2.0 * self.confusion_matrix.tp + self.confusion_matrix.fp + self.confusion_matrix.fn)

        self.values = self.dedupe_curve(tpr, f1_score_, maximize=True)


class TprMccMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_mcc_curve'
        super(TprMccMetric, self).__init__(**kwargs)

    def score(self):
        tpr = self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)
        matthews_correlation_coefficient = (self.confusion_matrix.tp * self.confusion_matrix.tn - self.confusion_matrix.fp * self.confusion_matrix.fn) /\
            ((self.confusion_matrix.fp + self.confusion_matrix.tp) * (self.confusion_matrix.tp + self.confusion_matrix.fn) *\
             (self.confusion_matrix.tn + self.confusion_matrix.fp) * (self.confusion_matrix.tn + self.confusion_matrix.fn))**0.5

        self.values = self.dedupe_curve(tpr, matthews_correlation_coefficient, maximize=True)


class TprInformednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_informedness_curve'
        super(TprInformednessMetric, self).__init__(**kwargs)

    def score(self):
        tpr = self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)
        tnr = self.confusion_matrix.tn / (self.confusion_matrix.fp + self.confusion_matrix.tn)

        informedness = tpr + tnr - 1

        self.values = self.dedupe_curve(tpr, informedness, maximize=True)


class TprMarkednessMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs['name'] = 'tpr_markedness_curve'
        super(TprMarkednessMetric, self).__init__(**kwargs)

    def score(self):
        tpr = self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)
        ppv = self.confusion_matrix.tp / (self.confusion_matrix.fp + self.confusion_matrix.tp)
        npv = self.confusion_matrix.tn / (self.confusion_matrix.tn + self.confusion_matrix.fn)

        markedness = ppv + npv - 1

        self.values = self.dedupe_curve(tpr, markedness, maximize=True)
