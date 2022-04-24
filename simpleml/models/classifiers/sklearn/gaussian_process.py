"""
Wrapper module around `sklearn.gaussian_process`
"""

__author__ = "Elisha Yadgaran"


from sklearn.gaussian_process import GaussianProcessClassifier

from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from .base_sklearn_classifier import SklearnClassifier

"""
Gaussian Process Classifier
"""


class WrappedSklearnGaussianProcessClassifier(
    GaussianProcessClassifier, ClassificationExternalModelMixin
):
    pass


class SklearnGaussianProcessClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnGaussianProcessClassifier(**kwargs)
