'''
Helper module to track installed libraries
'''

__author__ = 'Elisha Yadgaran'

import simpleml
import sqlalchemy
import sqlalchemy_mixins
import numpy as np
import pandas as pd
import dill
import psycopg2
import sklearn

INSTALLED_LIBRARIES = {
    'simpleml': simpleml.__version__,
    'sqlalchemy': sqlalchemy.__version__,
    # 'sqlalchemy_mixins': sqlalchemy_mixins.__version__,
    'numpy': np.__version__,
    'pandas': pd.__version__,
    'dill': dill.__version__,
    'psycopg2': psycopg2.__version__,
    'sklearn': sklearn.__version__
}
