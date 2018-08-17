'''
Wrapper module around `sklearn.neural_network`
'''

from simpleml.models.base_model import BaseModel
from simpleml.models.classifiers.classification_mixin import ClassificationMixin
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.neural_network import MLPClassifier


__author__ = 'Elisha Yadgaran'


'''
Perceptron
'''

class WrappedSklearnMLPClassifier(MLPClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnMLPClassifier(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnMLPClassifier(**kwargs)
