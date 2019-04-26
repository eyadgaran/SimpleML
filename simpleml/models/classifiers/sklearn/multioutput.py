'''
Wrapper module around `sklearn.multioutput`
'''

__author__ = 'Elisha Yadgaran'


from .base_sklearn_classifier import SklearnClassifier
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.multioutput import ClassifierChain, MultiOutputClassifier


'''
Classifier Chain
'''

class WrappedSklearnClassifierChain(ClassifierChain, ClassificationExternalModelMixin):
    pass

class SklearnClassifierChain(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnClassifierChain(**kwargs)


'''
Multi Output Classifier
'''

class WrappedSklearnMultiOutputClassifier(MultiOutputClassifier, ClassificationExternalModelMixin):
    pass

class SklearnMultiOutputClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnMultiOutputClassifier(**kwargs)
