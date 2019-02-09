'''
Wrapper module around `sklearn.naive_bayes`
'''

from .base_sklearn_classifier import SklearnClassifier
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB


__author__ = 'Elisha Yadgaran'


'''
Bernoulli
'''

class WrappedSklearnBernoulliNB(BernoulliNB, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnBernoulliNB(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnBernoulliNB(**kwargs)


'''
Gaussian
'''

class WrappedSklearnGaussianNB(GaussianNB, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnGaussianNB(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnGaussianNB(**kwargs)


'''
Multinomial
'''

class WrappedSklearnMultinomialNB(MultinomialNB, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnMultinomialNB(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnMultinomialNB(**kwargs)
