'''
Base module for keras classifier models
'''

__author__ = 'Elisha Yadgaran'


from simpleml.models.base_keras_model import KerasModel

import numpy as np


class KerasClassifier(KerasModel):
    def _predict(self, X):
        '''
        Keras returns class tuples (proba equivalent) so cast to single prediction
        '''
        return np.argmax(self.external_model.predict(X), axis=1)
