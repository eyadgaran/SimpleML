'''
Wrapper module around `sklearn.dummy`
'''

from simpleml.models.base_model import BaseModel
from simpleml.models.classifiers.classification_mixin import ClassificationMixin
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.dummy import DummyClassifier


__author__ = 'Elisha Yadgaran'


'''
Dummy classifier
'''

class WrappedSklearnDummyClassifier(DummyClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnDummyClassifier(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnDummyClassifier(**kwargs)
