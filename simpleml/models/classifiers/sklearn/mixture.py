'''
Wrapper module around `sklearn.mixture`
'''

__author__ = 'Elisha Yadgaran'


from .base_sklearn_classifier import SklearnClassifier
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.mixture import BayesianGaussianMixture, GaussianMixture


'''
Gaussian Mixture
'''

class WrappedSklearnBayesianGaussianMixture(BayesianGaussianMixture, ClassificationExternalModelMixin):
    pass

class SklearnBayesianGaussianMixture(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnBayesianGaussianMixture(**kwargs)


class WrappedSklearnGaussianMixture(GaussianMixture, ClassificationExternalModelMixin):
    pass

class SklearnGaussianMixture(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnGaussianMixture(**kwargs)
