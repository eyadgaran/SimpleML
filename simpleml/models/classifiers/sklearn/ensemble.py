'''
Wrapper module around `sklearn.ensemble`
'''

from simpleml.models.base_model import BaseModel
from simpleml.models.classifiers.classification_mixin import ClassificationMixin
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

class SklearnAdaBoostClassifier(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnAdaBoostClassifier(**kwargs)


'''
Bagging Classfier
'''

class WrappedSklearnBaggingClassifier(BaggingClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnBaggingClassifier(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnBaggingClassifier(**kwargs)


'''
Extra Trees Classifier
'''

class WrappedSklearnExtraTreesClassifier(ExtraTreesClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnExtraTreesClassifier(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnExtraTreesClassifier(**kwargs)


'''
Gradient Boosting Classifier
'''

class WrappedSklearnGradientBoostingClassifier(GradientBoostingClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnGradientBoostingClassifier(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnGradientBoostingClassifier(**kwargs)


'''
Random Forest Classifier
'''

class WrappedSklearnRandomForestClassifier(RandomForestClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnRandomForestClassifier(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnRandomForestClassifier(**kwargs)


'''
Voting Classifier
'''

class WrappedSklearnVotingClassifier(VotingClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnVotingClassifier(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnVotingClassifier(**kwargs)
