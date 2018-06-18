from simpleml.models.external_models import DefaultExternalModel


__author__ = 'Elisha Yadgaran'


class DefaultClassificationExternalModel(DefaultExternalModel):
    '''
    Wrapper class for a pickleable model with expected methods

    Expected to be used as Mixin Class with default methods and
    ovewritten by the model class if methods exist

    Extended from base class to add classification methods

    ex:

    from some_model_library import ActualModelClass

    class SomeClassificationExternalModel(ActualModelClass, DefaultClassificationExternalModel):
        pass

    class SimpleMLSomeClassificationExternalModel(BaseModel, [optional mixins]):
        def _create_external_model(self, **kwargs):
            return SomeClassificationExternalModel(**kwargs)
    '''

    def predict_proba(self, *args, **kwargs):
        '''
        By default nothing is implemented
        '''
        raise NotImplementedError
