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


# 3) Import optional dependencies or set to none/type to avoid import errors
# - Psycopg2
# - Keras
# - Hickle
import warnings
warning_msg = 'Unable to import optional dependency: {dependency}, to use install with `pip install {dependency}`'

try:
    import psycopg2
except ImportError:
    psycopg2 = None
    warnings.warn(warning_msg.format(dependency='psycopg2'), ImportWarning)

try:
    import keras
    from keras.models import Sequential, Model, load_model
except ImportError:
    keras = None
    load_model = None
    Sequential = type
    Model = type
    warnings.warn(warning_msg.format(dependency='keras'), ImportWarning)

try:
    import hickle
except ImportError:
    hickle = None
    warnings.warn(warning_msg.format(dependency='hickle'), ImportWarning)



# 4) Create simpleml local file directories
from . import utils


# 5) Import modules to register class names in global registry
from . import datasets
from . import pipelines
from . import models
from . import metrics
