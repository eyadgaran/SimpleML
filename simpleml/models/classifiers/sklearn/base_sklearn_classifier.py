'''
Base module for sklearn classifier models
'''

__author__ = 'Elisha Yadgaran'


from simpleml.models.base_sklearn_model import SklearnModel
from simpleml.models.classifiers.classification_mixin import ClassificationMixin


class SklearnClassifier(SklearnModel, ClassificationMixin):
    pass
