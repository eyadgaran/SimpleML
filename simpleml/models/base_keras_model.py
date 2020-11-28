'''
Base module for keras models. Keras has a native persistence mechanism so
need to overwrite other methods at the root
'''

__author__ = 'Elisha Yadgaran'


from simpleml.constants import TRAIN_SPLIT, VALIDATION_SPLIT
from .base_model import LibraryModel

import logging
from abc import abstractmethod


LOGGER = logging.getLogger(__name__)


class KerasModel(LibraryModel):
    def __init__(self,
                 use_training_generator=False, training_generator_params=None,
                 use_validation_generator=False, validation_generator_params=None,
                 use_sequence_object=False,
                 **kwargs):
        '''
        Pass default save method as Keras's persistence pattern

        :param use_training_generator: Whether to propagate use of a generator object
            when training -- does not allow for using a generator in production -- only fit_generator
        :type use_training_generator: Bool
        :param use_validation_generator: Whether to ALSO use a generator for validation
            data while training. Does nothing if use_training_generator is false
        :type use_validation_generator: Bool
        :param training_generator_params: parameters to pass to the generator method for train split -
            normal fit(_generator) params should be passed as params={}
        :param validation_generator_params: parameters to pass to the generator method for validation split -
            normal fit(_generator) params should be passed as params={}
        '''
        # Overwrite default model save pattern to keras specific (if not already passed)
        if 'save_patterns' not in kwargs:
            LOGGER.info('Setting model save pattern to `disk_keras_hdf5`')
            kwargs['save_patterns'] = {'model': ['disk_keras_hdf5']}
        elif 'model' not in kwargs['save_patterns']:
            LOGGER.info('Setting model save pattern to `disk_keras_hdf5`')
            kwargs['save_patterns']['model'] = ['disk_keras_hdf5']
        super(KerasModel, self).__init__(**kwargs)

        # Keras supports training and validation with generators
        # Design choice to put this in config as opposed to state because while
        # it is true that a specific combination of generator params will yield
        # the same model artifact as the traditional fit, it is very unlikely and
        # therefore assumed to be different (hashes will not be equal because of differing param structure)
        if training_generator_params is None:
            training_generator_params = {}
        if validation_generator_params is None:
            validation_generator_params = {}
        generator_params = {
            'use_training_generator': use_training_generator,
            'use_sequence_object': use_sequence_object,
            'training_generator_params': training_generator_params,
            'use_validation_generator': use_validation_generator,
            'validation_generator_params': validation_generator_params,
        }
        self.config.update(generator_params)

    @abstractmethod
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

    def build_network(self, external_model, **kwargs):
        '''
        Design choice to require build network method instead of exposing raw
        Keras objects that can be modified later. Simplifies saving and loading
        pattern because initialized object should also be the final state (as
        long as manual override doesnt happen)
        '''
        return external_model

    def _fit(self):
        '''
        Keras fit parameters (epochs, callbacks...) are stored as self.params so
        retrieve them automatically
        '''

        # Keras supports fitting on generator objects, so expose additional internal
        # method, if flag set
        if self.config['use_training_generator']:
            self._fit_generator()
        else:
            # Explicitly fit only on default (train) split
            split = self.transform(X=None, dataset_split=TRAIN_SPLIT, return_generator=False, return_sequence=False)
            # Hack for python <3.5 -- cant use fit(**split, **kwargs)
            temp_kwargs = self.get_params().copy()
            temp_kwargs.update(split)
            self.external_model.fit(**temp_kwargs)

    def _fit_generator(self):
        '''
        Keras fit parameters (epochs, callbacks...) are stored as self.params so
        retrieve them automatically
        '''
        use_keras_sequence = self.config.get('use_sequence_object', False)

        # Explicitly fit only on default (train) split
        training_generator = self.transform(X=None, dataset_split=TRAIN_SPLIT,
                                            return_generator=True,
                                            return_sequence=use_keras_sequence,
                                            **self.config.get('training_generator_params', {}))
        if self.config['use_validation_generator']:
            validation = self.transform(X=None, dataset_split=VALIDATION_SPLIT,
                                        return_generator=True,
                                        return_sequence=use_keras_sequence,
                                        **self.config.get('validation_generator_params', {}))
        else:
            validation = None

        self.external_model.fit_generator(
            training_generator, validation_data=validation, **self.get_params())

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
        if self.fitted:
            LOGGER.warning('Cannot change fit params and retrain model, skipping operation')
            LOGGER.warning('Initialize a new model for new fit params')
            return None

        self.params = kwargs

    def get_params(self, **kwargs):
        '''
        Get fit params
        '''
        return self.params

    @staticmethod
    def transfer_weights(new_model, old_model):
        new_layers = {i.name: i for i in new_model.layers}
        old_layers = {i.name: i for i in old_model.layers}

        for name, layer in new_layers.items():
            if name in old_layers:
                layer.set_weights(old_layers[name].get_weights())
