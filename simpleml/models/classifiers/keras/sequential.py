'''
Uses Keras's API to create a sequential classifier
'''

__author__ = 'Elisha Yadgaran'

from simpleml.models.classifiers.keras.base_keras_classifier import KerasClassifier
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

# Import optional dependencies
from simpleml.imports import Sequential


class WrappedKerasSequentialClassifier(Sequential, ClassificationExternalModelMixin):
    pass


class KerasSequentialClassifier(KerasClassifier):
    def _create_external_model(self, **kwargs):
        external_model = WrappedKerasSequentialClassifier(**kwargs)
        return self.build_network(external_model, **kwargs)
