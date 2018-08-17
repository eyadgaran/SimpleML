'''
Wrapper module around `sklearn.multioutput`
'''

from simpleml.models.base_model import BaseModel
from simpleml.models.classifiers.classification_mixin import ClassificationMixin
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.multioutput import ClassifierChain, MultiOutputClassifier


__author__ = 'Elisha Yadgaran'


'''
Classifier Chain
'''

class WrappedSklearnClassifierChain(ClassifierChain, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnClassifierChain(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnClassifierChain(**kwargs)


'''
Multi Output Classifier
'''

class WrappedSklearnMultiOutputClassifier(MultiOutputClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnMultiOutputClassifier(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnMultiOutputClassifier(**kwargs)
