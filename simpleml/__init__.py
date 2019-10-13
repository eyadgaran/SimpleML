'''
Startup module on initial import
'''

__author__ = 'Elisha Yadgaran'


# 1) Export package version
import pkg_resources

__version__ = pkg_resources.get_distribution(__name__).version


# 2) Load configs
from . import utils


# 3) Import modules to register class names in global registry
from . import datasets
from . import pipelines
from . import models
from . import metrics
