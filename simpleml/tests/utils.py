'''
Testing Utilities
'''

__author__ = 'Elisha Yadgaran'

import pandas as pd
import numpy as np

from typing import Any, Union
from pandas.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_array_equal
from simpleml.datasets.dataset_splits import Split
from simpleml.pipelines.projected_splits import ProjectedDatasetSplit


def assert_data_container_equal(a: Any, b: Any):
    '''
    Dynamic wrapper for custom equality functions
    '''
    if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
        assert_frame_equal(a, b)
    elif isinstance(a, pd.Series) or isinstance(b, pd.Series):
        assert_series_equal(a, b)
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        assert_array_equal(a, b)
    elif isinstance(a, (list, int, dict, str, type(None))) or isinstance(b, (list, int, dict, str, type(None))):
        assert(a == b)
    else:
        raise NotImplementedError


def assert_split_equal(a: Union[Split, ProjectedDatasetSplit], b: Union[Split, ProjectedDatasetSplit]) -> None:
    '''
    Util to compare split objects for equivalence
    '''
    assert(isinstance(a, (Split, ProjectedDatasetSplit)))
    assert(isinstance(b, (Split, ProjectedDatasetSplit)))

    assert(sorted(list(a.keys())) == sorted(list(b.keys())))

    for k, v in a.items():
        assert_data_container_equal(v, b[k])
