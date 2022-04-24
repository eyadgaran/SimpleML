'''
Wrapper module around `sklearn.neighbors`
'''

__author__ = 'Elisha Yadgaran'


from sklearn.neighbors import KNeighborsClassifier

from simpleml.models.classifiers.external_models import ClassificationExternalModelMixin

from .base_sklearn_classifier import SklearnClassifier

'''
K Neighbors
'''

class WrappedSklearnKNeighborsClassifier(KNeighborsClassifier, ClassificationExternalModelMixin):
    pass

class SklearnKNeighborsClassifier(SklearnClassifier):
    def _create_external_model(self, **kwargs):
        return WrappedSklearnKNeighborsClassifier(**kwargs)
