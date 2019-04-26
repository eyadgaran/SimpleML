'''
Wrapper module around `sklearn.neural_network`
'''

__author__ = 'Elisha Yadgaran'


from .base_sklearn_classifier import SklearnClassifier
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.neural_network import MLPClassifier


'''
Perceptron
'''

class WrappedSklearnMLPClassifier(MLPClassifier, ClassificationExternalModelMixin):
    pass

class SklearnMLPClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnMLPClassifier(**kwargs)
