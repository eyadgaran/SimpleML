"""
Module for split transfer objects. Internal object to shuttle dataset slices
between persistables
"""

__author__ = "Elisha Yadgaran"

from collections import defaultdict
from typing import Any

import pandas as pd

from simpleml.imports import ddDataFrame, ddSeries


class Split(dict):
    """
    Container class for splits
    """

    def __getattr__(self, attr):
        """
        Default attribute processor
        (Used in combination with __getitem__ to enable ** syntax)
        """
        return self.get(attr, None)

    @staticmethod
    def is_null_type(obj: Any) -> bool:
        """
        Helper to check for nulls - useful to not pass "empty" attributes
        so defaults of None will get returned downstream instead
        ex: **split -> all non null named params
        """
        # NoneType
        if obj is None:
            return True

        # Pandas objects
        if isinstance(obj, (pd.DataFrame, pd.Series)) and obj.empty:
            return True

        # Dask objects
        if isinstance(obj, (ddDataFrame, ddSeries)) and len(obj.index) == 0:
            return True

        # Empty built-ins - uses __nonzero__
        if isinstance(obj, (list, tuple, dict)) and not obj:
            return True

        # Else
        return False

    def squeeze(self):
        """
        Helper method to clear up any null-type keys
        """
        poppable_keys = [k for k, v in self.items() if self.is_null_type(v)]
        [self.pop(k) for k in poppable_keys]

        # Return self for easy chaining
        return self


class SplitContainer(defaultdict):
    """
    Explicit instantiation of a defaultdict returning split objects
    """

    def __init__(self, default_factory=Split, **kwargs):
        super(SplitContainer, self).__init__(default_factory, **kwargs)
