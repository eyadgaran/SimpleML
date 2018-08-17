'''
Wrapper module around `sklearn.naive_bayes`
'''

from simpleml.models.base_model import BaseModel
from simpleml.models.classifiers.classification_mixin import ClassificationMixin
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB


__author__ = 'Elisha Yadgaran'


'''
Bernoulli
'''

class WrappedSklearnBernoulliNB(BernoulliNB, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnBernoulliNB(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnBernoulliNB(**kwargs)


'''
Gaussian
'''

class WrappedSklearnGaussianNB(GaussianNB, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnGaussianNB(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnGaussianNB(**kwargs)


'''
Multinomial
'''

class WrappedSklearnMultinomialNB(MultinomialNB, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        pass

class SklearnMultinomialNB(BaseModel, ClassificationMixin):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnMultinomialNB(**kwargs)
