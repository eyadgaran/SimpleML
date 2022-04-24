"""
Wrapper module around `sklearn.multiclass`
"""

__author__ = "Elisha Yadgaran"


from sklearn.multiclass import (
    OneVsOneClassifier,
    OneVsRestClassifier,
    OutputCodeClassifier,
)

from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from .base_sklearn_classifier import SklearnClassifier

"""
One Vs Rest Classifier
"""


class WrappedSklearnOneVsRestClassifier(
    OneVsRestClassifier, ClassificationExternalModelMixin
):
    pass


class SklearnOneVsRestClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnOneVsRestClassifier(**kwargs)


"""
One Vs One Classifier
"""


class WrappedSklearnOneVsOneClassifier(
    OneVsOneClassifier, ClassificationExternalModelMixin
):
    pass


class SklearnOneVsOneClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnOneVsOneClassifier(**kwargs)


"""
Output Code Classifier
"""


class WrappedSklearnOutputCodeClassifier(
    OutputCodeClassifier, ClassificationExternalModelMixin
):
    pass


class SklearnOutputCodeClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnOutputCodeClassifier(**kwargs)
