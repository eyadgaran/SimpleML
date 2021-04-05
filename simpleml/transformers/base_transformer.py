__author__ = 'Elisha Yadgaran'


from sklearn.base import TransformerMixin as SklearnTransformerMixin
from typing import Any, Optional, Dict, List


class TransformerMixin(SklearnTransformerMixin):
    '''
    Base Transformer class that implements all the necessary methods

    Default behavior is to do nothing - overwrite later
    '''
    object_type: str = 'TRANSFORMER'

    def fit(self, X: Any, y: Optional[Any] = None, **kwargs):
        return self

    def transform(self, X: Any, y: Optional[Any] = None, **kwargs) -> Any:
        return X

    def get_params(self, **kwargs) -> Dict[str, Any]:
        '''
        Should only return seeding parameters, not fit ones
        (ie params of unfit object should be identical to fit object)
        '''
        return {}

    def set_params(self, **kwargs) -> None:
        pass

    def get_feature_names(self, input_feature_names: List[str]) -> List[str]:
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
        self.params: Dict[str, Any] = kwargs

    def get(self, param: str) -> Any:
        return self.params.get(param)

    def get_params(self, **kwargs) -> Dict[str, Any]:
        '''
        Should only return seeding parameters, not fit ones
        (ie params of unfit object should be identical to fit object)
        '''
        return self.params

    def set_params(self, **kwargs) -> None:
        self.params.update(kwargs)
