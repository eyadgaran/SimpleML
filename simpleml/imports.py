'''
Module to centralize all external imports - makes it easy to handle
optional dependencies in different installations
'''

__author__ = 'Elisha Yadgaran'


# Import optional dependencies or set to none/type to avoid import errors
# - Psycopg2
# - Keras
# - Hickle
# - Onedrivesdk
# - sshtunnel
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
    from keras.utils import Sequence
except ImportError:
    keras = None
    load_model = None
    Sequential = type
    Model = type
    Sequence = type
    warnings.warn(warning_msg.format(dependency='keras'), ImportWarning)

try:
    import hickle
except ImportError:
    hickle = None
    warnings.warn(warning_msg.format(dependency='hickle'), ImportWarning)

try:
    import onedrivesdk
except ImportError:
    onedrivesdk = None
    warnings.warn(warning_msg.format(dependency='onedrivesdk'), ImportWarning)

try:
    from sshtunnel import SSHTunnelForwarder
except ImportError:
    SSHTunnelForwarder = None
    warnings.warn(warning_msg.format(dependency='sshtunnel'), ImportWarning)
