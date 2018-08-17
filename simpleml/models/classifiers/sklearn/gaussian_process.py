'''
Wrapper module around `sklearn.gaussian_process`
'''

from simpleml.models.base_model import BaseModel
from simpleml.models.classifiers.classification_mixin import ClassificationMixin
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.gaussian_process import GaussianProcessClassifier


__author__ = 'Elisha Yadgaran'


'''
Gaussian Process Classifier
'''

class WrappedSklearnGaussianProcessClassifier(GaussianProcessClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnGaussianProcessClassifier(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnGaussianProcessClassifier(**kwargs)
