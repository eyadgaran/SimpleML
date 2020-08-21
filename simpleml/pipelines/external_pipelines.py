'''
Wrapper class for a pickleable pipeline of a series of transformers
'''

from collections import OrderedDict
from sklearn.pipeline import Pipeline

__author__ = 'Elisha Yadgaran'


class DefaultPipeline(OrderedDict):
    '''
    Use default dictionary behavior but add wrapper methods for
    extended functionality
    '''

    def add_transformer(self, name, transformer):
        '''
        Setter method for new transformer step
        '''
        self[name] = transformer

    def remove_transformer(self, name):
        '''
        Delete method for transformer step
        '''
        del self[name]

    def fit(self, X, y=None, **kwargs):
        '''
        Iterate through each transformation step and apply fit
        '''
        for step, transformer in self.items():
            X = transformer.fit_transform(X, y=y, **kwargs)

        return self

    def transform(self, X, **kwargs):
        '''
        Iterate through each transformation step and apply transform
        '''
        for step, transformer in self.items():
            X = transformer.transform(X, **kwargs)

        return X

    def fit_transform(self, X, y=None, **kwargs):
        '''
        Iterate through each transformation step and apply fit and transform
        '''
        for step, transformer in self.items():
            X = transformer.fit_transform(X, y=y, **kwargs)

        return X

    def get_params(self, params_only=None, **kwargs):
        '''
        Iterate through transformers and return parameters

        :param params_only: Unused parameter to align signature with Sklearn version
        '''
        params = {}
        for step, transformer in self.items():
            params[step] = transformer.get_params(**kwargs)

        return params

    def set_params(self, **params):
        '''
        Set params for transformers. Input is expected to be dict of dict

        :param params: dictionary of dictionaries. each dictionary must map to
        a transformer step
        '''
        for step, param in params.items():
            self[step].set_params(**param)

    def get_transformers(self):
        '''
        Get list of (step, transformer) tuples
        '''
        return [(i, j.__class__.__name__) for i, j in self.items()]

    def get_feature_names(self, feature_names):
        '''
        Iterate through each transformer and return list of resulting features
        starts with empty list by default but can pass in dataset as starting
        point to guide transformations

        :param feature_names: list of initial feature names before transformations
        :type: list
        '''
        for step, transformer in self.items():
            feature_names = transformer.get_feature_names(feature_names)

        return feature_names


class SklearnPipeline(Pipeline):
    '''
    Use default sklearn behavior but add wrapper methods for
    extended functionality
    '''

    def add_transformer(self, name, transformer, index=None):
        '''
        Setter method for new transformer step
        '''
        if index is not None:
            self.steps.insert(index, (name, transformer))
        else:
            self.steps.append((name, transformer))

    def remove_transformer(self, name):
        '''
        Delete method for transformer step
        '''
        index = [i for i, j in enumerate(self.steps) if j[0] == name][0]
        self.steps.pop(index)

    def get_params(self, params_only=False, **kwargs):
        '''
        Wrapper around sklearn implementation to drop non parameter returns
        :param params_only: boolean to filter down to actual transformer parameters
        '''
        params = super(SklearnPipeline, self).get_params(**kwargs)

        if params_only:
            # actual params have k__v format
            steps = params.pop('steps', [])
            step_names = [step[0] for step in steps]
            return {k: v for k, v in params.items() if k not in step_names}
        else:
            return params

    def get_transformers(self):
        '''
        Get list of (step, transformer) tuples
        '''
        return [(i, j.__class__.__name__) for i, j in self.steps]

    def get_feature_names(self, feature_names):
        '''
        Iterate through each transformer and return list of resulting features
        starts with empty list by default but can pass in dataset as starting
        point to guide transformations

        :param feature_names: list of initial feature names before transformations
        :type: list
        '''
        for step, transformer in self.steps:
            feature_names = transformer.get_feature_names(feature_names)

        return feature_names
