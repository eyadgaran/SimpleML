'''
Seq2Seq Keras classifiers
'''

__author__ = 'Elisha Yadgaran'

from simpleml.models.classifiers.keras.model import KerasModelClassifier

from abc import abstractmethod
import numpy as np


class KerasSeq2SeqClassifier(KerasModelClassifier):
    '''
    Base class for sequence to sequence models. Differ from traditional models
    because training and inference use different architectures
    '''
    @abstractmethod
    def build_inference_network(self, model):
        '''
        Inference network - Differs from training one so gets established dynamically
        at inference time

        return: inference_model(s)
        rtype: self.external_model.__class__
        '''

    def check_for_models(self, rebuild=False):
        if not hasattr(self, 'inference_model'):
            self.inference_model = self.build_inference_network(self.external_model.__class__)
            self.transfer_weights(new_model=self.inference_model, old_model=self.external_model)
        elif rebuild:
            self.transfer_weights(new_model=self.inference_model, old_model=self.external_model)

    def _predict(self, X):
        '''
        Inference network differs from training one so gets established dynamically
        at inference time. Does NOT get persisted since the weights are duplicative
        to the training ones. And the training network can in theory be updated
        with new training data later
        '''
        self.check_for_models()
        return self.inference_model.predict(X)


class KerasEncoderDecoderClassifier(KerasSeq2SeqClassifier):
    def check_for_models(self, rebuild=False):
        if not hasattr(self, 'encoder_model') or not hasattr(self, 'decoder_model'):
            self.encoder_model, self.decoder_model = self.build_inference_network(self.external_model.__class__)
            self.transfer_weights(new_model=self.encoder_model, old_model=self.external_model)
            self.transfer_weights(new_model=self.decoder_model, old_model=self.external_model)
        elif rebuild:
            self.transfer_weights(new_model=self.encoder_model, old_model=self.external_model)
            self.transfer_weights(new_model=self.decoder_model, old_model=self.external_model)

    def _predict(self, X, start_index, end_index, max_length=None, **kwargs):
        '''
        Inference network differs from training one so gets established dynamically
        at inference time. Does NOT get persisted since the weights are duplicative
        to the training ones. And the training network can in theory be updated
        with new training data later

        Runs full encoder/decoder loop
        1) Encodes input into initial decoder state
        2) Loops through decoder state until:
            - End token is predicted
            - Max length is reached
        '''
        self.check_for_models()

        # Encode the input as state vectors for the decoder.
        decoder_states = self.encode(X)
        if not isinstance(decoder_states, list):  # GRU only returns single state vs lstm h, c
            decoder_states = [decoder_states]

        # Generate empty target sequence of length 1 with the start index
        # Assumes sparse integer representation (not one hot encoding)
        predicted_sequence = np.ones((X.shape[0], 1)) * start_index

        while True:
            decoder_output_and_states = self.decode([predicted_sequence] + decoder_states)
            decoder_prediction = decoder_output_and_states[0]
            decoder_states = decoder_output_and_states[1:]  # Multiple for lstm, single for gru

            # Next tokens
            next_tokens = np.argmax(decoder_prediction, axis=1).reshape(-1, 1)
            predicted_sequence = np.concatenate([predicted_sequence, next_tokens], axis=1)

            # Exit conditions
            # TODO: generic support for different length sequences
            if (next_tokens == end_index).any() or (max_length is not None and predicted_sequence.shape[1] >= max_length):
                break

        return predicted_sequence

    def encode(self, X):
        self.check_for_models()

        # Run through encoder model and return encoder state to condition decoder
        return self.encoder_model.predict(X)

    def decode(self, X):
        self.check_for_models()

        # Run through decoder model and return prediction and next state
        return self.decoder_model.predict(X)
