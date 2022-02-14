'''
Import modules to register class names in global registry
'''

__author__ = 'Elisha Yadgaran'


from simpleml.utils.errors import DatasetError

from .base_dataset import Dataset
from . import dask
from . import pandas
from . import numpy
