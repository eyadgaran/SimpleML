'''
Utility to inspect supported params
'''

__author__ = 'Elisha Yadgaran'


import inspect
import logging
from typing import Any, Callable, Dict, Mapping

LOGGER = logging.getLogger(__name__)


def signature_kwargs_validator(fn: Callable, **kwargs) -> Dict[str, Any]:
    '''
    Takes a function and arbitrary kwargs. Returns the set that match or everything
    if function takes arbitrary kwargs
    '''
    supported_kwargs = {}
    signature_params = inspect.signature(fn).parameters

    # check if any params are **kwargs (all inputs accepted)
    has_kwarg_params = any([param.kind == param.VAR_KEYWORD for param in signature_params.values()])
    # log ignored args
    if not has_kwarg_params:
        for arg, val in kwargs.items():
            if arg not in signature_params:
                LOGGER.warning(f'Unsupported param encountered, `{arg}`. Dropping...')
            else:
                supported_kwargs[arg] = val
    else:
        supported_kwargs = kwargs

    return supported_kwargs
