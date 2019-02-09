'''
Base module for keras classifier models
'''

__author__ = 'Elisha Yadgaran'


from simpleml.models.base_keras_model import KerasModel
from simpleml.models.classifiers.classification_mixin import ClassificationMixin

import numpy as np


class KerasClassifier(KerasModel, ClassificationMixin):
    def _predict(self, X, **kwargs):
        '''
        Keras returns class tuples (proba equivalent) so cast to single prediction
        '''
        return np.argmax(self.external_model.predict(X), axis=1)
