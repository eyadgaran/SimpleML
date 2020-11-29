'''
Base Module for all external models

Inherit and extend for particular models
'''

__author__ = 'Elisha Yadgaran'


from simpleml.registries import KerasRegistry
from future.utils import with_metaclass


class ExternalModelMixin(with_metaclass(KerasRegistry, object)):
    '''
    Wrapper class for a pickleable model with expected methods

    Expected to be used as Mixin Class with default methods and
    ovewritten by the model class if methods exist

    ex:

    from some_model_library import ActualModelClass

    class WrappedActualModelClass(ActualModelClass, ExternalModelMixin):
        pass

    class some_model_libraryActualModelClass(Model, [optional mixins]):
        def _create_external_model(self, **kwargs):
            return WrappedActualModelClass(**kwargs)
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
        return {}

    def set_params(self, *args, **kwargs):
        '''
        By default nothing is implemented
        '''

    def score(self, *args, **kwargs):
        '''
        By default nothing is implemented
        '''
        raise NotImplementedError

    def get_feature_metadata(self, *args, **kwargs):
        '''
        By default nothing is implemented
        '''
        return None
