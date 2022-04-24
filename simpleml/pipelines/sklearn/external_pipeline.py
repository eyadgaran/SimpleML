"""
External pipeline support for scikit-learn pipeline
"""

__author__ = "Elisha Yadgaran"


from typing import Any, Dict, List, Optional, Tuple

from sklearn.pipeline import Pipeline

from simpleml.pipelines.external_pipelines import ExternalPipelineMixin


class SklearnExternalPipeline(Pipeline, ExternalPipelineMixin):
    """
    wrap sklearn pipeline with standardized methods
    """

    def add_transformer(
        self, name: str, transformer: Any, index: Optional[int] = None
    ) -> None:
        """
        Setter method for new transformer step
        """
        if index is not None:
            self.steps.insert(index, (name, transformer))
        else:
            self.steps.append((name, transformer))

    def remove_transformer(self, name: str) -> None:
        """
        Delete method for transformer step
        """
        index = [i for i, j in enumerate(self.steps) if j[0] == name][0]
        self.steps.pop(index)

    def get_params(self, params_only: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Wrapper around sklearn implementation to drop non parameter returns
        :param params_only: boolean to filter down to actual transformer parameters
        """
        params = super(SklearnExternalPipeline, self).get_params(**kwargs)

        if params_only:
            # actual params have k__v format
            steps = params.pop("steps", [])
            step_names = [step[0] for step in steps]
            return {k: v for k, v in params.items() if k not in step_names}
        else:
            return params

    def get_transformers(self) -> List[Tuple[str, str]]:
        """
        Get list of (step, transformer) tuples
        """
        return [(i, j.__class__.__name__) for i, j in self.steps]

    def get_feature_names(self, feature_names: List[str]) -> List[str]:
        """
        Iterate through each transformer and return list of resulting features
        starts with empty list by default but can pass in dataset as starting
        point to guide transformations

        :param feature_names: list of initial feature names before transformations
        :type: list
        """
        for step, transformer in self.steps:
            feature_names = transformer.get_feature_names(feature_names)

        return feature_names
