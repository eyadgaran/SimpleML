'''
Base module for keras models. Keras has a native persistence mechanism so
need to overwrite other methods at the root
'''

__author__ = 'Elisha Yadgaran'


import logging
from simpleml.models.base_model import BaseModel


LOGGER = logging.getLogger(__name__)


class BaseKerasModel(BaseModel):
    def __init__(self, save_method='disk_keras_hdf5', **kwargs):
        '''
        Pass default save method as Keras's persistence pattern
        '''
        super(BaseKerasModel, self).__init__(save_method=save_method, **kwargs)

    def _create_external_model(self, **kwargs):
        '''
        Abstract method for each subclass to implement
        should return the desired model object

        Must return external_file

        Keras pattern is:
        external_model = SomeWrappedKerasClass(**kwargs)
        return self.build_network(external_model)
        '''
        external_model = None
        self.build_network(external_model, **kwargs)

        raise NotImplementedError

    def build_network(self, external_model, **kwargs):
        '''
        Design choice to require build network method instead of exposing raw
        Keras objects that can be modified later. Simplifies saving and loading
        pattern because initialized object should also be the final state (as
        long as manual override doesnt happen)
        '''
        return external_model

    def _fit(self, X, y):
        '''
        Keras fit parameters (epochs, callbacks...) are stored as self.params so
        retrieve them automatically
        '''

        self.external_model.fit(X, y, **self.get_params())

    def _predict(self, X):
        '''
        Keras returns class tuples (proba equivalent) so cast to single prediction
        '''
        # Keras by default supports proba predictions, so coerce to integers
        return self.external_model.predict(X).round()

    def set_params(self, **kwargs):
        '''
        Keras networks don't have params beyond layers, which should be configured
        in `self.build_network`, so use this for fit params - self.fit will auto pull
        params and pass them to the fit method.

        TODO: Figure out if changing params should be allowed after fit. If they are,
        would need to reinitialize model, otherwise it would train more epochs and not
        forget the original training. If not, once fit, we can treat the model as
        static, and no longer able to be changed

        For now going with option 2 - cannot refit models
        '''
        if self.state.get('fitted'):
            LOGGER.warning('Cannot change fit params and retrain model, skipping operation')
            LOGGER.warning('Initialize a new model for new fit params')
            return None

        self.params = kwargs

    def get_params(self, **kwargs):
        '''
        Get fit params
        '''
        return self.params
