"""
Import modules to register class names in global registry
"""

__author__ = "Elisha Yadgaran"


from simpleml.utils.errors import DatasetError

from . import dask, numpy, pandas
from .base_dataset import Dataset
