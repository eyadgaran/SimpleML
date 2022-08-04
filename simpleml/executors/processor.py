"""
Entry for execution
"""

__author__ = "Elisha Yadgaran"


import logging
from typing import Callable

from simpleml.registries import EXECUTOR_REGISTRY

LOGGER = logging.getLogger(__name__)


class AbstractBaseExecutor(object):
    @staticmethod
    def process(op, *args, **kwargs):
        raise NotImplementedError


class ExecutionProcessor(object):
    """
    Core class to process execution requests
    """

    @staticmethod
    def process(op: Callable, *args, executor: str = None, msg: str = None, **kwargs):
        # infer executor from environment
        configured_executor: str = EXECUTOR_REGISTRY.get("default")
        if executor is not None and executor != configured_executor:
            LOGGER.warning(
                f"Default executor explicitly overwritten from {configured_executor} to {executor}. {msg}"
            )

        executor: AbstractBaseExecutor = EXECUTOR_REGISTRY.get(
            executor or configured_executor
        )
        return executor.process(op=op, *args, **kwargs)
