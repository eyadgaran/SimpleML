'''
Wrapper module around `sklearn.multioutput`
'''

from .base_sklearn_classifier import SklearnClassifier
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.multioutput import ClassifierChain, MultiOutputClassifier


__author__ = 'Elisha Yadgaran'


'''
Classifier Chain
'''

class WrappedSklearnClassifierChain(ClassifierChain, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnClassifierChain(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnClassifierChain(**kwargs)


'''
Multi Output Classifier
'''

class WrappedSklearnMultiOutputClassifier(MultiOutputClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnMultiOutputClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnMultiOutputClassifier(**kwargs)
