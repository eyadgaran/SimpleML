'''
Wrapper module around `sklearn.tree`
'''

__author__ = 'Elisha Yadgaran'


import logging

from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from .base_sklearn_classifier import SklearnClassifier

LOGGER = logging.getLogger(__name__)

'''
Trees
'''

class WrappedSklearnDecisionTreeClassifier(DecisionTreeClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        feature_importances = self.feature_importances_.squeeze()
        if features is None or len(features) < len(feature_importances):
            LOGGER.warning('Fewer feature names than features passed, defaulting to numbered list')
            features = range(len(feature_importances))
        return dict(zip(features, feature_importances))

class SklearnDecisionTreeClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnDecisionTreeClassifier(**kwargs)


class WrappedSklearnExtraTreeClassifier(ExtraTreeClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        feature_importances = self.feature_importances_.squeeze()
        if features is None or len(features) < len(feature_importances):
            LOGGER.warning('Fewer feature names than features passed, defaulting to numbered list')
            features = range(len(feature_importances))
        return dict(zip(features, feature_importances))

class SklearnExtraTreeClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnExtraTreeClassifier(**kwargs)
