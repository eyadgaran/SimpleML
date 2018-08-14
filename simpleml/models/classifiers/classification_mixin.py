from simpleml.utils.errors import ModelError


__author__ = 'Elisha Yadgaran'


class ClassificationMixin(object):
    '''
    Mixin class for classification methods
    '''
    def predict_proba(self, X, **kwargs):
        '''
        Pass through method to external model after running through pipeline
        '''
        if not self.state['fitted']:
            raise ModelError('Must fit model before predicting')

        transformed = self.pipeline.transform(X, **kwargs)

        return self.external_model.predict_proba(transformed)
