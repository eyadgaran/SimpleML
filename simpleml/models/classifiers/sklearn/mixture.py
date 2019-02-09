'''
Wrapper module around `sklearn.mixture`
'''

from .base_sklearn_classifier import SklearnClassifier
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.mixture import BayesianGaussianMixture, GaussianMixture


__author__ = 'Elisha Yadgaran'


'''
Gaussian Mixture
'''

class WrappedSklearnBayesianGaussianMixture(BayesianGaussianMixture, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnBayesianGaussianMixture(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnBayesianGaussianMixture(**kwargs)


class WrappedSklearnGaussianMixture(GaussianMixture, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnGaussianMixture(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnGaussianMixture(**kwargs)
