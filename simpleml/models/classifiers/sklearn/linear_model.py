'''
Wrapper module around `sklearn.linear_model`
'''

from simpleml.models.base_model import Model
from simpleml.models.classifiers.classification_mixin import ClassificationMixin
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Perceptron,\
    RidgeClassifier, RidgeClassifierCV, SGDClassifier


__author__ = 'Elisha Yadgaran'


'''
Logistic Regression
'''

class WrappedSklearnLogisticRegression(LogisticRegression, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnLogisticRegression(Model, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnLogisticRegression(**kwargs)

class WrappedSklearnLogisticRegressionCV(LogisticRegressionCV, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnLogisticRegressionCV(Model, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnLogisticRegressionCV(**kwargs)


'''
Perceptron
'''

class WrappedSklearnPerceptron(Perceptron, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnPerceptron(Model, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnPerceptron(**kwargs)


'''
Ridge Classifier
'''

class WrappedSklearnRidgeClassifier(RidgeClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnRidgeClassifier(Model, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnRidgeClassifier(**kwargs)

class WrappedSklearnRidgeClassifierCV(RidgeClassifierCV, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnRidgeClassifierCV(Model, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnRidgeClassifierCV(**kwargs)


'''
SGD Classifier
'''

class WrappedSklearnSGDClassifier(SGDClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnSGDClassifier(Model, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnSGDClassifier(**kwargs)
