'''
Wrapper module around `sklearn.dummy`
'''

from .base_sklearn_classifier import SklearnClassifier
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.dummy import DummyClassifier


__author__ = 'Elisha Yadgaran'


'''
Dummy classifier
'''

class WrappedSklearnDummyClassifier(DummyClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnDummyClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnDummyClassifier(**kwargs)
