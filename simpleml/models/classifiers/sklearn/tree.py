'''
Wrapper module around `sklearn.tree`
'''

from simpleml.models.base_model import BaseModel
from simpleml.models.classifiers.classification_mixin import ClassificationMixin
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier


__author__ = 'Elisha Yadgaran'


'''
Trees
'''

class WrappedSklearnDecisionTreeClassifier(DecisionTreeClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnDecisionTreeClassifier(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnDecisionTreeClassifier(**kwargs)


class WrappedSklearnExtraTreeClassifier(ExtraTreeClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnExtraTreeClassifier(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnExtraTreeClassifier(**kwargs)
