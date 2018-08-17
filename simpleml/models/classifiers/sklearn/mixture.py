'''
Wrapper module around `sklearn.mixture`
'''

from simpleml.models.base_model import BaseModel
from simpleml.models.classifiers.classification_mixin import ClassificationMixin
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.mixture import BayesianGaussianMixture, GaussianMixture


__author__ = 'Elisha Yadgaran'


'''
Gaussian Mixture
'''

class WrappedSklearnBayesianGaussianMixture(BayesianGaussianMixture, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnBayesianGaussianMixture(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnBayesianGaussianMixture(**kwargs)


class WrappedSklearnGaussianMixture(GaussianMixture, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnGaussianMixture(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnGaussianMixture(**kwargs)
