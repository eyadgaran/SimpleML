"""
Base native python pipeline wrapper
"""

__author__ = "Elisha Yadgaran"


import logging
from typing import Any, List

from simpleml.pipelines.base_pipeline import Pipeline

from .external_pipeline import OrderedDictExternalPipeline

LOGGER = logging.getLogger(__name__)


class OrderedDictPipeline(Pipeline):
    """
    Native python dict pipeline implementation
    """

    def _create_external_pipeline(
        self, transformers: List[Any], **kwargs
    ) -> OrderedDictExternalPipeline:
        return OrderedDictExternalPipeline(transformers)
