from simpleml.models.external_models import ExternalModelMixin
import logging


__author__ = 'Elisha Yadgaran'


LOGGER = logging.getLogger(__name__)


class ClassificationExternalModelMixin(ExternalModelMixin):
    '''
    Wrapper class for a pickleable model with expected methods

    Expected to be used as Mixin Class with default methods and
    ovewritten by the model class if methods exist

    Extended from base class to add classification methods

    ex:

    from some_model_library import ActualModelClass

    class WrappedActualModelClass(ActualModelClass, ClassificationExternalModelMixin):
        pass

    class some_model_libraryActualModelClass(Model, [optional mixins]):
        def _create_external_model(self, **kwargs):
            return WrappedActualModelClass(**kwargs)
    '''

    def predict_proba(self, *args, **kwargs):
        '''
        By default fall back to predict method
        '''
        LOGGER.warning('No predict_proba method defined, using predict')
        return self.predict(*args, **kwargs)
