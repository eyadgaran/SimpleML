"""
Startup module on initial import
"""

__author__ = "Elisha Yadgaran"


# 1) Export package version
import pkg_resources

__version__ = pkg_resources.get_distribution(__name__).version


# 3) Import modules to register class names in global registry
# 2) Load configs
from . import datasets, metrics, models, pipelines, utils
