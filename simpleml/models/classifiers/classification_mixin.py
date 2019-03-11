from simpleml.utils.errors import ModelError
import numpy as np


__author__ = 'Elisha Yadgaran'


class ClassificationMixin(object):
    '''
    Mixin class for classification methods
    '''
    def predict_proba(self, X, transform=True, **kwargs):
        '''
        Pass through method to external model after running through pipeline
        :param transform: bool, whether to transform input via pipeline
         before predicting, default True
        '''
        if not self.state['fitted']:
            raise ModelError('Must fit model before predicting')

        if transform:
            X = self.pipeline.transform(X, **kwargs)

        if X is None:  # Don't attempt to run through model if no samples
            return np.array([])

        return self.external_model.predict_proba(X)
