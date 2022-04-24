"""
Wrapper module around `sklearn.svm`
"""

__author__ = "Elisha Yadgaran"


import logging

from sklearn.svm import SVC, LinearSVC, NuSVC

from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from .base_sklearn_classifier import SklearnClassifier

LOGGER = logging.getLogger(__name__)

"""
Support Vectors
"""


class WrappedSklearnLinearSVC(LinearSVC, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        # coefficients generated for each class >2, only report for class 0
        coefficients = self.coef_[0].squeeze()
        if features is None or len(features) < len(coefficients):
            LOGGER.warning(
                "Fewer feature names than features passed, defaulting to numbered list"
            )
            features = range(len(coefficients))
        return dict(zip(features, coefficients))


class SklearnLinearSVC(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnLinearSVC(**kwargs)


class WrappedSklearnNuSVC(NuSVC, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        # coefficients generated only for linear kernels.
        # Report support vectors instead
        support_vectors = self.support_vectors_[0].squeeze()
        if features is None or len(features) < len(support_vectors):
            LOGGER.warning(
                "Fewer feature names than features passed, defaulting to numbered list"
            )
            features = range(len(support_vectors))
        return dict(zip(features, support_vectors))


class SklearnNuSVC(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnNuSVC(**kwargs)


class WrappedSklearnSVC(SVC, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        # coefficients generated only for linear kernels.
        # Report support vectors instead
        support_vectors = self.support_vectors_[0].squeeze()
        if features is None or len(features) < len(support_vectors):
            LOGGER.warning(
                "Fewer feature names than features passed, defaulting to numbered list"
            )
            features = range(len(support_vectors))
        return dict(zip(features, support_vectors))


class SklearnSVC(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnSVC(**kwargs)
