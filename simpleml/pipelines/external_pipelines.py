'''
Base class for standardized pipeline methods
'''

__author__ = 'Elisha Yadgaran'


from typing import Any, Dict, List, Optional, Tuple


class ExternalPipelineMixin(object):
    '''
    Mixin to add unimplemented stubs for standardized api
    '''

    def add_transformer(self, name: str, transformer: Any) -> None:
        '''
        Setter method for new transformer step
        '''
        raise NotImplementedError

    def remove_transformer(self, name: str) -> None:
        '''
        Delete method for transformer step
        '''
        raise NotImplementedError

    def fit(self, X: Any, y: Optional[Any] = None, **kwargs):
        '''
        Iterate through each transformation step and apply fit
        '''
        raise NotImplementedError

    def reset(self) -> None:
        '''
        Command to reset any saved state in the transformers. Expects each
        transformer to implement if there is any partial state (e.g. partial_fit
        called)
        '''
        raise NotImplementedError

    def partial_fit(self, X: Any, y: Optional[Any] = None, **kwargs):
        '''
        Iterate through each transformation step and apply partial fit
        '''
        raise NotImplementedError

    def transform(self, X: Any, **kwargs) -> Any:
        '''
        Iterate through each transformation step and apply transform
        '''
        raise NotImplementedError

    def fit_transform(self, X: Any, y: Optional[Any] = None, **kwargs) -> Any:
        '''
        Iterate through each transformation step and apply fit and transform
        '''
        raise NotImplementedError

    def get_params(self, params_only: Optional[bool] = None, **kwargs) -> Dict[str, Any]:
        '''
        Iterate through transformers and return parameters

        :param params_only: Unused parameter to align signature with Sklearn version
        '''
        raise NotImplementedError

    def set_params(self, **params) -> None:
        '''
        Set params for transformers. Input is expected to be dict of dict

        :param params: dictionary of dictionaries. each dictionary must map to
        a transformer step
        '''
        raise NotImplementedError

    def get_transformers(self) -> List[Tuple[str, str]]:
        '''
        Get list of (step, transformer) tuples
        '''
        raise NotImplementedError

    def get_feature_names(self, feature_names: List[str]) -> List[str]:
        '''
        Iterate through each transformer and return list of resulting features
        starts with empty list by default but can pass in dataset as starting
        point to guide transformations

        :param feature_names: list of initial feature names before transformations
        :type: list
        '''
        raise NotImplementedError
