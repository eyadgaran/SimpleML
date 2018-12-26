'''
Uses Keras's API to create a model classifier
'''

__author__ = 'Elisha Yadgaran'

from simpleml.models.base_keras_model import BaseKerasModel
from simpleml.models.classifiers.classification_mixin import ClassificationMixin
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

# Import optional dependencies
from simpleml import Model


class WrappedKerasModelClassifier(Model, ClassificationExternalModelMixin):
    pass


class KerasModelClassifier(BaseKerasModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        external_model = WrappedKerasModelClassifier
        return self.build_network(external_model, **kwargs)
