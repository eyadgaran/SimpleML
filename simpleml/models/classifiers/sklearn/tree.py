'''
Wrapper module around `sklearn.tree`
'''

from .base_sklearn_classifier import SklearnClassifier
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


__author__ = 'Elisha Yadgaran'


'''
Trees
'''

class WrappedSklearnDecisionTreeClassifier(DecisionTreeClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnDecisionTreeClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnDecisionTreeClassifier(**kwargs)


class WrappedSklearnExtraTreeClassifier(ExtraTreeClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnExtraTreeClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnExtraTreeClassifier(**kwargs)
