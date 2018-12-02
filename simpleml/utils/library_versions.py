'''
Helper module to track installed libraries
'''

__author__ = 'Elisha Yadgaran'

import pkg_resources

def safe_lookup(package):
    try:
        return pkg_resources.get_distribution(package).version
    except pkg_resources.DistributionNotFound:
        return None


INSTALLED_LIBRARIES = {
    'simpleml': safe_lookup('simpleml'),
    'sqlalchemy': safe_lookup('sqlalchemy'),
    'sqlalchemy_mixins': safe_lookup('sqlalchemy_mixins'),
    'numpy': safe_lookup('numpy'),
    'pandas': safe_lookup('pandas'),
    'dill': safe_lookup('dill'),
    'psycopg2': safe_lookup('psycopg2'),
    'sklearn': safe_lookup('sklearn'),
    'hickle': safe_lookup('hickle'),
    'keras': safe_lookup('keras'),
    'tensorflow': safe_lookup('tensorflow')
}
