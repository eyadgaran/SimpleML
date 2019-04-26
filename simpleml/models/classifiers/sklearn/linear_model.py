'''
Wrapper module around `sklearn.linear_model`
'''

__author__ = 'Elisha Yadgaran'


from .base_sklearn_classifier import SklearnClassifier
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Perceptron,\
    RidgeClassifier, RidgeClassifierCV, SGDClassifier
import logging

LOGGER = logging.getLogger(__name__)

'''
Logistic Regression
'''

class WrappedSklearnLogisticRegression(LogisticRegression, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        coefficients = self.coef_.squeeze()
        if features is None or len(features) < len(coefficients):
            LOGGER.warning('Fewer feature names than features passed, defaulting to numbered list')
            features = range(len(coefficients))
        return dict(zip(features, coefficients))

class SklearnLogisticRegression(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnLogisticRegression(**kwargs)

class WrappedSklearnLogisticRegressionCV(LogisticRegressionCV, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        coefficients = self.coef_.squeeze()
        if features is None or len(features) < len(coefficients):
            LOGGER.warning('Fewer feature names than features passed, defaulting to numbered list')
            features = range(len(coefficients))
        return dict(zip(features, coefficients))

class SklearnLogisticRegressionCV(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnLogisticRegressionCV(**kwargs)


'''
Perceptron
'''

class WrappedSklearnPerceptron(Perceptron, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        coefficients = self.coef_.squeeze()
        if features is None or len(features) < len(coefficients):
            LOGGER.warning('Fewer feature names than features passed, defaulting to numbered list')
            features = range(len(coefficients))
        return dict(zip(features, coefficients))

class SklearnPerceptron(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnPerceptron(**kwargs)


'''
Ridge Classifier
'''

class WrappedSklearnRidgeClassifier(RidgeClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        coefficients = self.coef_.squeeze()
        if features is None or len(features) < len(coefficients):
            LOGGER.warning('Fewer feature names than features passed, defaulting to numbered list')
            features = range(len(coefficients))
        return dict(zip(features, coefficients))

class SklearnRidgeClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnRidgeClassifier(**kwargs)

class WrappedSklearnRidgeClassifierCV(RidgeClassifierCV, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        coefficients = self.coef_.squeeze()
        if features is None or len(features) < len(coefficients):
            LOGGER.warning('Fewer feature names than features passed, defaulting to numbered list')
            features = range(len(coefficients))
        return dict(zip(features, coefficients))

class SklearnRidgeClassifierCV(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnRidgeClassifierCV(**kwargs)


'''
SGD Classifier
'''

class WrappedSklearnSGDClassifier(SGDClassifier, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        coefficients = self.coef_.squeeze()
        if features is None or len(features) < len(coefficients):
            LOGGER.warning('Fewer feature names than features passed, defaulting to numbered list')
            features = range(len(coefficients))
        return dict(zip(features, coefficients))

class SklearnSGDClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnSGDClassifier(**kwargs)
