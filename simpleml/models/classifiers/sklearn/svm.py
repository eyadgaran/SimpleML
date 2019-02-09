'''
Wrapper module around `sklearn.svm`
'''

from .base_sklearn_classifier import SklearnClassifier
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.svm import LinearSVC, NuSVC, SVC


__author__ = 'Elisha Yadgaran'


'''
Support Vectors
'''

class WrappedSklearnLinearSVC(LinearSVC, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnLinearSVC(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnLinearSVC(**kwargs)


class WrappedSklearnNuSVC(NuSVC, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnNuSVC(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnNuSVC(**kwargs)


class WrappedSklearnSVC(SVC, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnSVC(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnSVC(**kwargs)
