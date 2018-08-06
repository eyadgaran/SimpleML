'''
Module for classification metrics

Includes base class and derived metrics following the nomenclature:
    ConstraintValueMetric
Where:
    Constraint is the lookup criteria (ex FPR in ROC curve)
    Value is desired value (ex TPR in ROC curve)
'''

from simpleml.metrics.base_metric import BaseMetric
from simpleml.pipelines.base_pipeline import TRAIN_SPLIT, VALIDATION_SPLIT
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


__author__ = 'Elisha Yadgaran'


class ClassificationMetric(BaseMetric):
    '''
    TODO: Figure out multiclass generalizations
    '''
    pass


class BinaryClassificationMetric(ClassificationMetric):
    def __init__(self, dataset_split=TRAIN_SPLIT, **kwargs):
        '''
        :param dataset_split: string denoting which dataset split to use
            can be one of: `TRAIN`, `VALIDATION`, Other. Other gets no prefix
            Default is train split to stay consistent with no split mapping to Train
            in BasePipeline

        '''
        name = kwargs.pop('name', '')
        self.dataset_split = dataset_split

        # Explicitly call out in sample or validation metrics
        if dataset_split == TRAIN_SPLIT:
            name = 'in_sample_' + name
        elif dataset_split == VALIDATION_SPLIT:
            name = 'validation_' + name

        super(BinaryClassificationMetric, self).__init__(name=name, **kwargs)

        # Initialize confusion matrix
        self._confusion_matrix = None

        # Thresholds to compute confusion matrix at (default every 0.005 increment)
        self.thresholds = np.linspace(0, 1, 201)

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
        probabilities = self.model.predict_proba(X=None, dataset_split=self.dataset_split)
        labels = self.model.get_labels(dataset_split=self.dataset_split)

        results = []
        for threshold in self.thresholds:
            predictions = np.where(probabilities[:, 1] >= threshold, 1, 0)
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


class FprTprMetric(BinaryClassificationMetric):
    def __init__(self, **kwargs):
        # Drop whatever name was passed and explicitly rename
        kwargs.pop('name', '')
        name = 'fpr_tpr_curve'
        super(FprTprMetric, self).__init__(name=name, **kwargs)

    def score(self):
        fpr = self.confusion_matrix.fp / (self.confusion_matrix.fp + self.confusion_matrix.tn)
        tpr = self.confusion_matrix.tp / (self.confusion_matrix.tp + self.confusion_matrix.fn)

        self.values = self.dedupe_curve(fpr, tpr)
