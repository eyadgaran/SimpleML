'''
Wrapper module around `sklearn.linear_model`
'''

from simpleml.models.base_model import BaseModel
from simpleml.models.classifiers.classification_mixin import ClassificationMixin
from simpleml.models.classifiers.external_models import DefaultClassificationExternalModel

from sklearn.linear_model import LogisticRegression


__author__ = 'Elisha Yadgaran'


'''
Logistic Regression
'''

class WrappedSklearnLogisticRegression(LogisticRegression, DefaultClassificationExternalModel):
    def get_feature_metadata(self, **kwargs):
        pass

class SklearnLogisticRegression(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnLogisticRegression(**kwargs)
