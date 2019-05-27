'''
Wrapper module around `sklearn.ensemble`
'''

__author__ = 'Elisha Yadgaran'


from .base_sklearn_classifier import SklearnClassifier
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,\
    GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
import logging

LOGGER = logging.getLogger(__name__)

'''
AdaBoost Classifier
'''

# TODO: experiment with try excepts on different estimator types (feature_importances_, coefficients, ...)

class WrappedSklearnAdaBoostClassifier(AdaBoostClassifier, ClassificationExternalModelMixin):
    # Boosted classifier doesnt have an easy way to aggregate feature metadata
    pass

class SklearnAdaBoostClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnAdaBoostClassifier(**kwargs)


'''
Bagging Classfier
'''

class WrappedSklearnBaggingClassifier(BaggingClassifier, ClassificationExternalModelMixin):
    # Bagging classifier doesnt have an easy way to aggregate feature metadata
    pass

class SklearnBaggingClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnBaggingClassifier(**kwargs)


'''
Extra Trees Classifier
'''

class WrappedSklearnExtraTreesClassifier(ExtraTreesClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        feature_importances = self.feature_importances_.squeeze()
        if features is None or len(features) < len(feature_importances):
            LOGGER.warning('Fewer feature names than features passed, defaulting to numbered list')
            features = range(len(feature_importances))
        return dict(zip(features, feature_importances))

class SklearnExtraTreesClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnExtraTreesClassifier(**kwargs)


'''
Gradient Boosting Classifier
'''

class WrappedSklearnGradientBoostingClassifier(GradientBoostingClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        feature_importances = self.feature_importances_.squeeze()
        if features is None or len(features) < len(feature_importances):
            LOGGER.warning('Fewer feature names than features passed, defaulting to numbered list')
            features = range(len(feature_importances))
        return dict(zip(features, feature_importances))

class SklearnGradientBoostingClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnGradientBoostingClassifier(**kwargs)


'''
Random Forest Classifier
'''

class WrappedSklearnRandomForestClassifier(RandomForestClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        feature_importances = self.feature_importances_.squeeze()
        if features is None or len(features) < len(feature_importances):
            LOGGER.warning('Fewer feature names than features passed, defaulting to numbered list')
            features = range(len(feature_importances))
        return dict(zip(features, feature_importances))

class SklearnRandomForestClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnRandomForestClassifier(**kwargs)


'''
Voting Classifier
'''

class WrappedSklearnVotingClassifier(VotingClassifier, ClassificationExternalModelMixin):
    # Voting classifier doesnt have an easy way to aggregate feature metadata
    pass

class SklearnVotingClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnVotingClassifier(**kwargs)
