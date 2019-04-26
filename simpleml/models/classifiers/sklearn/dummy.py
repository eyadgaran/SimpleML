'''
Wrapper module around `sklearn.dummy`
'''

__author__ = 'Elisha Yadgaran'


from .base_sklearn_classifier import SklearnClassifier
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.dummy import DummyClassifier


'''
Dummy classifier
'''

class WrappedSklearnDummyClassifier(DummyClassifier, ClassificationExternalModelMixin):
    # Dummy model doesnt have any feature metadata
    pass

class SklearnDummyClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnDummyClassifier(**kwargs)
