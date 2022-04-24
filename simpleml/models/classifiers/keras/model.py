"""
Uses Keras's API to create a model classifier
"""

__author__ = "Elisha Yadgaran"

# Import optional dependencies
from simpleml.imports import Model
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin
from simpleml.models.classifiers.keras.base_keras_classifier import KerasClassifier


class WrappedKerasModelClassifier(Model, ClassificationExternalModelMixin):
    pass


class KerasModelClassifier(KerasClassifier):
    def _create_external_model(self, **kwargs):
        external_model = WrappedKerasModelClassifier
        return self.build_network(external_model, **kwargs)
