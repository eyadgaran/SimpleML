'''
Wrapper module around `sklearn.multiclass`
'''

from simpleml.models.base_model import BaseModel
from simpleml.models.classifiers.classification_mixin import ClassificationMixin
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier, OutputCodeClassifier


__author__ = 'Elisha Yadgaran'


'''
One Vs Rest Classifier
'''

class WrappedSklearnOneVsRestClassifier(OneVsRestClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnOneVsRestClassifier(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnOneVsRestClassifier(**kwargs)


'''
One Vs One Classifier
'''

class WrappedSklearnOneVsOneClassifier(OneVsOneClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnOneVsOneClassifier(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnOneVsOneClassifier(**kwargs)


'''
Output Code Classifier
'''

class WrappedSklearnOutputCodeClassifier(OutputCodeClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnOutputCodeClassifier(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnOutputCodeClassifier(**kwargs)
