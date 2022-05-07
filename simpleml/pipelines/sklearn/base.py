"""
Base Sklearn pipeline wrapper
"""

__author__ = "Elisha Yadgaran"


import inspect
import logging
from typing import Any, Dict, List

from simpleml.pipelines.base_pipeline import Pipeline
from simpleml.pipelines.projected_splits import ProjectedDatasetSplit
from simpleml.utils.signature_inspection import signature_kwargs_validator

from .external_pipeline import SklearnExternalPipeline

LOGGER = logging.getLogger(__name__)


class SklearnPipeline(Pipeline):
    """
    Scikit-Learn Pipeline implementation
    """

    def _create_external_pipeline(
        self, transformers: List[Any], **kwargs
    ) -> SklearnExternalPipeline:
        """
        Initialize a scikit-learn pipeline object
        """
        supported_kwargs = signature_kwargs_validator(
            SklearnExternalPipeline.__init__, **kwargs
        )

        return SklearnExternalPipeline(
            transformers,
            # Only supported sklearn params
            **supported_kwargs,
        )

    def _filter_fit_params(self, split: ProjectedDatasetSplit) -> Dict[str, Any]:
        """
        Sklearn Pipelines register arbitrary input kwargs but validate non X,y
        as `stepname__parameter` format
        """
        supported_fit_params = {}

        # Ensure input compatibility with split object
        fit_params = inspect.signature(self.external_pipeline.fit).parameters
        for split_arg, val in split.items():
            if split_arg not in fit_params and "__" not in split_arg:
                LOGGER.warning(
                    f"Unsupported fit param encountered, `{split_arg}`. Dropping..."
                )
            else:
                supported_fit_params[split_arg] = val

        return supported_fit_params
