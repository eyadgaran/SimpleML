"""
External pipeline support for native python pipeline
"""

__author__ = "Elisha Yadgaran"


from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from simpleml.pipelines.external_pipelines import ExternalPipelineMixin


class OrderedDictExternalPipeline(OrderedDict, ExternalPipelineMixin):
    """
    Use default dictionary behavior but add wrapper methods for
    extended functionality
    """

    def add_transformer(self, name: str, transformer: Any) -> None:
        """
        Setter method for new transformer step
        """
        self[name] = transformer

    def remove_transformer(self, name: str) -> None:
        """
        Delete method for transformer step
        """
        del self[name]

    def fit(self, X: Any, y: Optional[Any] = None, **kwargs):
        """
        Iterate through each transformation step and apply fit
        """
        for step, transformer in self.items():
            X = transformer.fit_transform(X, y=y, **kwargs)

        return self

    def transform(self, X: Any, **kwargs) -> Any:
        """
        Iterate through each transformation step and apply transform
        """
        for step, transformer in self.items():
            X = transformer.transform(X, **kwargs)

        return X

    def fit_transform(self, X: Any, y: Optional[Any] = None, **kwargs) -> Any:
        """
        Iterate through each transformation step and apply fit and transform
        """
        for step, transformer in self.items():
            X = transformer.fit_transform(X, y=y, **kwargs)

        return X

    def get_params(
        self, params_only: Optional[bool] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Iterate through transformers and return parameters

        :param params_only: Unused parameter to align signature with Sklearn version
        """
        params = {}
        for step, transformer in self.items():
            params[step] = transformer.get_params(**kwargs)

        return params

    def set_params(self, **params) -> None:
        """
        Set params for transformers. Input is expected to be dict of dict

        :param params: dictionary of dictionaries. each dictionary must map to
        a transformer step
        """
        for step, param in params.items():
            self[step].set_params(**param)

    def get_transformers(self) -> List[Tuple[str, str]]:
        """
        Get list of (step, transformer) tuples
        """
        return [(i, j.__class__.__name__) for i, j in self.items()]

    def get_feature_names(self, feature_names: List[str]) -> List[str]:
        """
        Iterate through each transformer and return list of resulting features
        starts with empty list by default but can pass in dataset as starting
        point to guide transformations

        :param feature_names: list of initial feature names before transformations
        :type: list
        """
        for step, transformer in self.items():
            feature_names = transformer.get_feature_names(feature_names)

        return feature_names
