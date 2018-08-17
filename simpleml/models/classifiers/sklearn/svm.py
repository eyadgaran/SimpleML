'''
Wrapper module around `sklearn.svm`
'''

from simpleml.models.base_model import BaseModel
from simpleml.models.classifiers.classification_mixin import ClassificationMixin
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.svm import LinearSVC, NuSVC, SVC


__author__ = 'Elisha Yadgaran'


'''
Support Vectors
'''

class WrappedSklearnLinearSVC(LinearSVC, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnLinearSVC(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnLinearSVC(**kwargs)


class WrappedSklearnNuSVC(NuSVC, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnNuSVC(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnNuSVC(**kwargs)


class WrappedSklearnSVC(SVC, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnSVC(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnSVC(**kwargs)
