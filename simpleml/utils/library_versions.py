"""
Helper module to track installed libraries
"""

__author__ = "Elisha Yadgaran"

import sys
from typing import Dict

import pkg_resources


def safe_lookup(package):
    try:
        return pkg_resources.get_distribution(package).version
    except pkg_resources.DistributionNotFound:
        return None


INSTALLED_LIBRARIES: Dict[str, str] = {
    "python": "{}.{}.{}".format(*sys.version_info[:3]),
    "simpleml": safe_lookup("simpleml"),
    "sqlalchemy": safe_lookup("sqlalchemy"),
    "sqlalchemy_mixins": safe_lookup("sqlalchemy_mixins"),
    "numpy": safe_lookup("numpy"),
    "pandas": safe_lookup("pandas"),
    "cloudpickle": safe_lookup("cloudpickle"),
    "psycopg2": safe_lookup("psycopg2"),
    "scikit-learn": safe_lookup("scikit-learn"),
    "hickle": safe_lookup("hickle"),
    "tensorflow": safe_lookup("tensorflow"),
    "scipy": safe_lookup("scipy"),
    "sqlalchemy_json": safe_lookup("sqlalchemy-json"),
    "onedrivesdk": safe_lookup("onedrivesdk"),
    "sshtunnel": safe_lookup("sshtunnel"),
    "apache-libcloud": safe_lookup("apache-libcloud"),
    "dask": safe_lookup("dask"),
}
