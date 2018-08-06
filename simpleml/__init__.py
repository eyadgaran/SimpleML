# Import utils on load to create expected directories
import utils
from _version import __version__
import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO
)


__author__ = 'Elisha Yadgaran'
