'''
Wrapper module around `sklearn.multiclass`
'''

from .base_sklearn_classifier import SklearnClassifier
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier


__author__ = 'Elisha Yadgaran'


'''
One Vs Rest Classifier
'''

class WrappedSklearnOneVsRestClassifier(OneVsRestClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnOneVsRestClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnOneVsRestClassifier(**kwargs)


'''
One Vs One Classifier
'''

class WrappedSklearnOneVsOneClassifier(OneVsOneClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnOneVsOneClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnOneVsOneClassifier(**kwargs)


'''
Output Code Classifier
'''

class WrappedSklearnOutputCodeClassifier(OutputCodeClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnOutputCodeClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnOutputCodeClassifier(**kwargs)
