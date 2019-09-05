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
        self.assert_fitted('Must fit model before predicting')

        if transform:
            # Pipeline returns Split object if input is null
            # Otherwise transformed matrix
            transformed = self.transform(X, **kwargs)
            X = transformed.X if X is None else transformed

        if X is None:  # Don't attempt to run through model if no samples (can't evaulate ahead of transform in case dataset split used)
            return np.array([])

        return self.external_model.predict_proba(X)
