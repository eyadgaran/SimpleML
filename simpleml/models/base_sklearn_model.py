'''
Base module for Sklearn models.
'''

__author__ = 'Elisha Yadgaran'


from .base_model import Model


class SklearnModel(Model):
    '''
    No different than base model. Here just to maintain the pattern
    Generic Base -> Library Base -> Domain Base -> Individual Models
    (ex: Model -> SklearnModel -> SklearnClassifier -> SklearnLogisticRegression)
    '''
    pass
