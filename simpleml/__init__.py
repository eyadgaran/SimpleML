'''
Startup module on initial import
'''

__author__ = 'Elisha Yadgaran'


# 1) Configure logging
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)


# 2) Export package version
import pkg_resources

__version__ = pkg_resources.get_distribution(__name__).version


# 3) Load configs
from . import utils


# 4) Import modules to register class names in global registry
from . import datasets
from . import pipelines
from . import models
from . import metrics
