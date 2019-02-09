'''
Wrapper module around `sklearn.linear_model`
'''

from .base_sklearn_classifier import SklearnClassifier
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

class SklearnLogisticRegression(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnLogisticRegression(**kwargs)

class WrappedSklearnLogisticRegressionCV(LogisticRegressionCV, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnLogisticRegressionCV(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnLogisticRegressionCV(**kwargs)


'''
Perceptron
'''

class WrappedSklearnPerceptron(Perceptron, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnPerceptron(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnPerceptron(**kwargs)


'''
Ridge Classifier
'''

class WrappedSklearnRidgeClassifier(RidgeClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnRidgeClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnRidgeClassifier(**kwargs)

class WrappedSklearnRidgeClassifierCV(RidgeClassifierCV, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnRidgeClassifierCV(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnRidgeClassifierCV(**kwargs)


'''
SGD Classifier
'''

class WrappedSklearnSGDClassifier(SGDClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnSGDClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnSGDClassifier(**kwargs)
