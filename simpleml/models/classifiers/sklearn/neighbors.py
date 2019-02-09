'''
Wrapper module around `sklearn.neighbors`
'''

from .base_sklearn_classifier import SklearnClassifier
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.neighbors import KNeighborsClassifier


__author__ = 'Elisha Yadgaran'


'''
K Neighbors
'''

class WrappedSklearnKNeighborsClassifier(KNeighborsClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnKNeighborsClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnKNeighborsClassifier(**kwargs)
