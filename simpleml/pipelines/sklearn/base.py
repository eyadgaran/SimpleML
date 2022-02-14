'''
Base Sklearn pipeline wrapper
'''

__author__ = 'Elisha Yadgaran'


import inspect
import logging
from typing import Any, Dict, List

from simpleml.pipelines.base_pipeline import Pipeline
from simpleml.pipelines.projected_splits import ProjectedDatasetSplit

from .external_pipeline import SklearnExternalPipeline

LOGGER = logging.getLogger(__name__)


class SklearnPipeline(Pipeline):
    '''
    Scikit-Learn Pipeline implementation
    '''

    def _create_external_pipeline(self,
                                  transformers: List[Any],
                                  **kwargs) -> SklearnExternalPipeline:
        '''
        Initialize a scikit-learn pipeline object
        '''
        # Ensure input compatibility with split object
        init_params = inspect.signature(SklearnExternalPipeline.__init__).parameters
        # check if any params are **kwargs (all inputs accepted)
        has_kwarg_params = any([param.kind == param.VAR_KEYWORD for param in init_params.values()])
        # log ignored args
        if not has_kwarg_params:
            supported_kwargs = {k: v for k, v in kwargs.items() if k in init_params}
        else:
            supported_kwargs = kwargs

        return SklearnExternalPipeline(
            transformers,
            # Only supported sklearn params
            **supported_kwargs
        )

    def _filter_fit_params(self, split: ProjectedDatasetSplit) -> Dict[str, Any]:
        '''
        Sklearn Pipelines register arbitrary input kwargs but validate non X,y
        as `stepname__parameter` format
        '''
        supported_fit_params = {}

        # Ensure input compatibility with split object
        fit_params = inspect.signature(self.external_pipeline.fit).parameters
        for split_arg, val in split.items():
            if split_arg not in fit_params and '__' not in split_arg:
                LOGGER.warning(f'Unsupported fit param encountered, `{split_arg}`. Dropping...')
            else:
                supported_fit_params[split_arg] = val

        return supported_fit_params
