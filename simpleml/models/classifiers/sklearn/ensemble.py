'''
Wrapper module around `sklearn.ensemble`
'''

from .base_sklearn_classifier import SklearnClassifier
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,\
    GradientBoostingClassifier, RandomForestClassifier, VotingClassifier


__author__ = 'Elisha Yadgaran'


'''
AdaBoost Classifier
'''

class WrappedSklearnAdaBoostClassifier(AdaBoostClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnAdaBoostClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnAdaBoostClassifier(**kwargs)


'''
Bagging Classfier
'''

class WrappedSklearnBaggingClassifier(BaggingClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnBaggingClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnBaggingClassifier(**kwargs)


'''
Extra Trees Classifier
'''

class WrappedSklearnExtraTreesClassifier(ExtraTreesClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnExtraTreesClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnExtraTreesClassifier(**kwargs)


'''
Gradient Boosting Classifier
'''

class WrappedSklearnGradientBoostingClassifier(GradientBoostingClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnGradientBoostingClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnGradientBoostingClassifier(**kwargs)


'''
Random Forest Classifier
'''

class WrappedSklearnRandomForestClassifier(RandomForestClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnRandomForestClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnRandomForestClassifier(**kwargs)


'''
Voting Classifier
'''

class WrappedSklearnVotingClassifier(VotingClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnVotingClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnVotingClassifier(**kwargs)
