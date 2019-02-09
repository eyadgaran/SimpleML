'''
Wrapper module around `sklearn.neural_network`
'''

from .base_sklearn_classifier import SklearnClassifier
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.neural_network import MLPClassifier


__author__ = 'Elisha Yadgaran'


'''
Perceptron
'''

class WrappedSklearnMLPClassifier(MLPClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnMLPClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnMLPClassifier(**kwargs)
