from sklearn.base import TransformerMixin as SklearnTransformerMixin


__author__ = 'Elisha Yadgaran'


class TransformerMixin(SklearnTransformerMixin):
    '''
    Base Transformer class that implements all the necessary methods

    Default behavior is to do nothing - overwrite later
    '''
    object_type = 'TRANSFORMER'

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        return X

    def get_params(self, **kwargs):
        '''
        Should only return seeding parameters, not fit ones
        (ie params of unfit object should be identical to fit object)
        '''
        return {}

    def set_params(self, **kwargs):
        pass

    def get_feature_names(self, input_feature_names):
        return input_feature_names


class Transformer(TransformerMixin):
    '''
    Base Transformer class with param management - Can interfere with mro
    if used as a mixin - Use `TransformerMixin` in that case
    '''
    def __init__(self, **kwargs):
        '''
        Assumes only seeding kwargs passed - will affect hash otherwise
        if random unused parameters are passed
        '''
        self.params = kwargs

    def get(self, param):
        return self.params.get(param)

    def get_params(self, **kwargs):
        '''
        Should only return seeding parameters, not fit ones
        (ie params of unfit object should be identical to fit object)
        '''
        return self.params

    def set_params(self, **kwargs):
        self.params.update(kwargs)
