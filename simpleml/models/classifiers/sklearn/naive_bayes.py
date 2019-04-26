'''
Wrapper module around `sklearn.naive_bayes`
'''

__author__ = 'Elisha Yadgaran'


from .base_sklearn_classifier import SklearnClassifier
from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
import logging

LOGGER = logging.getLogger(__name__)

'''
Bernoulli
'''

class WrappedSklearnBernoulliNB(BernoulliNB, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        # By default probabilities are returned for all classes, only displays class 0
        log_probs = self.feature_log_prob_.squeeze()[0]
        if features is None or len(features) < len(log_probs):
            LOGGER.warning('Fewer feature names than features passed, defaulting to numbered list')
            features = range(len(log_probs))
        return dict(zip(features, log_probs))

class SklearnBernoulliNB(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnBernoulliNB(**kwargs)


'''
Gaussian
'''

class WrappedSklearnGaussianNB(GaussianNB, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        # By default probabilities are returned for all classes, only displays class 0
        thetas = self.theta_.squeeze()[0]
        if features is None or len(features) < len(thetas):
            LOGGER.warning('Fewer feature names than features passed, defaulting to numbered list')
            features = range(len(thetas))
        return dict(zip(features, thetas))

class SklearnGaussianNB(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnGaussianNB(**kwargs)


'''
Multinomial
'''

class WrappedSklearnMultinomialNB(MultinomialNB, ClassificationExternalModelMixin):
    def get_feature_metadata(self, features, **kwargs):
        # By default probabilities are returned for all classes, only displays class 0
        log_probs = self.feature_log_prob_.squeeze()[0]
        if features is None or len(features) < len(log_probs):
            LOGGER.warning('Fewer feature names than features passed, defaulting to numbered list')
            features = range(len(log_probs))
        return dict(zip(features, log_probs))

class SklearnMultinomialNB(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnMultinomialNB(**kwargs)
