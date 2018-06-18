'''
Base Module for all external models

Inherit and extend for particular models
'''

__author__ = 'Elisha Yadgaran'


class DefaultExternalModel(object):
    '''
    Wrapper class for a pickleable model with expected methods

    Expected to be used as Mixin Class with default methods and
    ovewritten by the model class if methods exist

    ex:

    from some_model_library import ActualModelClass

    class SomeExternalModel(ActualModelClass, DefaultExternalModel):
        pass

    class SimpleMLSomeExternalModel(BaseModel, [optional mixins]):
        def _create_external_model(self, **kwargs):
            return SomeExternalModel(**kwargs)
    '''

    def fit(self, *args, **kwargs):
        '''
        By default nothing is implemented
        '''
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        '''
        By default nothing is implemented
        '''
        raise NotImplementedError

    def get_params(self, *args, **kwargs):
        '''
        By default nothing is implemented
        '''
        raise NotImplementedError

    def set_params(self, *args, **kwargs):
        '''
        By default nothing is implemented
        '''
        raise NotImplementedError

    def score(self, *args, **kwargs):
        '''
        By default nothing is implemented
        '''
        raise NotImplementedError

    def get_feature_metadata(self, *args, **kwargs):
        '''
        By default nothing is implemented
        '''
        raise NotImplementedError
